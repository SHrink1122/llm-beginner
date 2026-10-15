import io
import json
import math
import re
import sys
import time
import zipfile
from pathlib import Path

import swanlab
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.chat import build_labels, format_messages   
from src.lora import inject_lora, save_lora       

MODEL_DIR = ROOT / "models" / "Qwen2.5-0.5B"          
DATA_DIR = ROOT / "data" / "moss-sft"                 
CKPT_DIR = ROOT / "ckpt" / "sft"                      
TARGET_MODULES = ["q_proj", "v_proj"]                 

LIMIT = 10000
EPOCHS = 3
BATCH_SIZE = 1 # 显存8GB只能跑batchsize为1
GRAD_ACCUM = 16      # 等效 batch = BATCH_SIZE * GRAD_ACCUM
LR = 2e-4
MAX_LEN = 2048
LORA_R = 8
LORA_ALPHA = 16

SWANLAB_PROJECT = "qwen2.5-sft-dpo"  
SWANLAB_RUN = "sft"                 


def clean(text):
    # 删moss标记
    return re.compile(r"<\|[^|>]+?\|>:\s*|<eo[a-z]>").sub("", text).strip()


def to_messages(rec):
    msgs = [{"role": "system", "content": rec["meta_instruction"]}]
    for turn in rec["chat"].values():
        if not turn.get("Human") or not turn.get("MOSS"):
            continue
        msgs.append({"role": "user", "content": clean(turn["Human"])})
        msgs.append({"role": "assistant", "content": clean(turn["MOSS"])})
    return msgs


def iter_records(limit):
    # 流式读取
    zips = sorted(DATA_DIR.glob("*.jsonl.zip"))
    if zips:
        zf = zipfile.ZipFile(zips[0])
        f = io.TextIOWrapper(zf.open(zf.namelist()[0]), encoding="utf-8")
    else:
        f = open(sorted(DATA_DIR.glob("*.jsonl"))[0], encoding="utf-8")
    with f:  
        for i, line in enumerate(f):
            if limit and i >= limit:     # limit=0 时读全量
                break
            yield json.loads(line)       


def encode(rec, tokenizer, max_len):
    msgs = to_messages(rec)
    if len(msgs) < 3:          
        return None              
    text = format_messages(msgs)    
    ids = tokenizer(text).input_ids[:max_len] 
    labels = build_labels(ids, msgs)[:max_len].tolist()

    if all(l == -100 for l in labels):
        return None # 整体mask跳过 防止nan
    return {"input_ids": ids, "labels": labels}


def collate(batch, pad_id):
    width = max(len(s["input_ids"]) for s in batch)
    ids, attn, labels = [], [], []
    for s in batch:
        pad = width - len(s["input_ids"])
        ids.append(s["input_ids"] + [pad_id] * pad)             
        attn.append([1] * len(s["input_ids"]) + [0] * pad)       # attn mask
        labels.append(s["labels"] + [-100] * pad)                # CrossEntropyLoss 
    return torch.tensor(ids), torch.tensor(attn), torch.tensor(labels)


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 本地跑 bf16 省显存
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=dtype)
    model.config.use_cache = False 
    inject_lora(model, TARGET_MODULES, r=LORA_R, alpha=LORA_ALPHA)
    # 本地运行5060 8GB 显存过小 采取梯度累积训练
    if device == "cuda":
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    model.to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    total = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in trainable)
    print(f"设备 {device} / {dtype}，可训参数 {trainable_num:,} "
          f"({trainable_num / total:.2%})") 

    samples = []
    for rec in iter_records(LIMIT):
        s = encode(rec, tokenizer, MAX_LEN)
        if s:
            samples.append(s)
    print(f"样本 {len(samples)} 条，平均 {sum(len(s['input_ids']) for s in samples) / len(samples):.0f} token")

    loader = DataLoader(samples, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=lambda b: collate(b, tokenizer.pad_token_id))
    opt = torch.optim.AdamW(trainable, lr=LR)

    swanlab.init(
        project=SWANLAB_PROJECT,
        name=f"{SWANLAB_RUN}-{time.strftime('%m%d-%H%M%S')}",
        config={
            "base_model": MODEL_DIR.name,
            "target_modules": TARGET_MODULES,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "limit": LIMIT,
            "samples": len(samples),
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM,
            "batch_size_effective": BATCH_SIZE * GRAD_ACCUM,
            "steps_per_epoch": math.ceil(len(loader) / GRAD_ACCUM),
            "lr": LR,
            "max_len": MAX_LEN,
            "trainable_params": trainable_num,
            "trainable_ratio": round(trainable_num / total, 6),
            "device": device,
            "dtype": str(dtype),
        },
    )

    run = swanlab.get_run()
    run.define_metric("epoch/loss", x_axis="epoch/step", section_name="epoch")
    run.define_metric("epoch/step", hidden=True, section_name="epoch") 

    model.train()
    global_step = 0
    print("[训练中] loss 曲线见 swanlab")
    for epoch in range(EPOCHS):
        epoch_loss, epoch_tok = 0.0, 0
        win_loss, win_n = 0.0, 0
        opt.zero_grad()
        for i, (ids, attn, labels) in enumerate(loader):
            loss = model(input_ids=ids.to(device), attention_mask=attn.to(device),
                         labels=labels.to(device)).loss
            (loss / GRAD_ACCUM).backward()
            win_loss += loss.item()
            win_n += 1
            n_tok = int((labels != -100).sum())
            epoch_loss += loss.item() * n_tok
            epoch_tok += n_tok

            if (i + 1) % GRAD_ACCUM == 0 or i + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                opt.zero_grad()
                global_step += 1
                swanlab.log({"train/loss": win_loss / win_n}, step=global_step)
                win_loss, win_n = 0.0, 0

        if epoch_tok:
            swanlab.log(
                {"epoch/step": epoch + 1, "epoch/loss": epoch_loss / epoch_tok},
                step=global_step,
            )

    save_lora(model, CKPT_DIR, r=LORA_R, alpha=LORA_ALPHA, target_modules=TARGET_MODULES,
              base_model=str(MODEL_DIR.name), samples=len(samples), epochs=EPOCHS)
    print(f"LoRA 权重已存到 {CKPT_DIR}")


if __name__ == "__main__":
    main()
