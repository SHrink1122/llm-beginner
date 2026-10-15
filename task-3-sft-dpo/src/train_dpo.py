import json
import math
import sys
import time
from pathlib import Path

import swanlab
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))           

from src.chat import build_labels, format_messages   
from src.lora import inject_lora, load_lora, save_lora   

MODEL_DIR = ROOT / "models" / "Qwen2.5-0.5B"  
SFT_DIR = ROOT / "ckpt" / "sft"                
CKPT_DIR = ROOT / "ckpt" / "dpo"               
DATA_FILE = ROOT / "data" / "dpo-enzh20k" / "dpo_zh.json"
ROLE = {"human": "user", "gpt": "assistant"}  

LIMIT = 1000           
EPOCHS = 2
BATCH_SIZE = 2    # 一个偏好对 两个side     
GRAD_ACCUM = 16         
LR = 5e-5               
BETA = 0.1             
MAX_LEN = 1024          

SWANLAB_PROJECT = "qwen2.5-sft-dpo"   
SWANLAB_RUN = "dpo"


def to_messages(conversations):
    return [{"role": ROLE.get(c["from"], c["from"]), "content": c["value"]}
            for c in conversations]


def encode(rec, tokenizer, max_len):
    prompt = to_messages(rec["conversations"])
    pair = {}
    for side in ("chosen", "rejected"):
        resp = rec[side]["value"].strip()
        if not resp:                     # 空回答整条样本丢掉
            return None
        msgs = prompt + [{"role": "assistant", "content": resp}]
        ids = tokenizer(format_messages(msgs)).input_ids[:max_len]
        labels = build_labels(ids, msgs)[:max_len].tolist()
        if all(l == -100 for l in labels):
            return None
        pair[side] = (ids, labels)
    return pair


def collate(pairs, side, pad_id):
    batch = [p[side] for p in pairs]
    width = max(len(ids) for ids, _ in batch)  
    ids_, attn, labels = [], [], []
    for ids, lab in batch:
        pad = width - len(ids)
        ids_.append(ids + [pad_id] * pad)
        attn.append([1] * len(ids) + [0] * pad)
        labels.append(lab + [-100] * pad)  
    return torch.tensor(ids_), torch.tensor(attn), torch.tensor(labels)


def seq_logp(model, ids, attn, labels):
    logits = model(input_ids=ids, attention_mask=attn).logits[:, :-1].float()
    tgt = labels[:, 1:]
    logp = F.log_softmax(logits, dim=-1)                       # [B, L-1, V]
    logp = logp.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (logp * (tgt != -100)).sum(-1)


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    cfg = json.loads((SFT_DIR / "config.json").read_text(encoding="utf-8"))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    def build_sft_model(grad_ckpt=False):
        m = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=dtype)
        m.config.use_cache = False          
        inject_lora(m, cfg["target_modules"], r=cfg["r"], alpha=cfg["alpha"])
        load_lora(m, SFT_DIR)               
        if grad_ckpt:
            m.gradient_checkpointing_enable()
            m.enable_input_require_grads() 
        return m.to(device)

    model = build_sft_model(grad_ckpt=device == "cuda")  
    ref = build_sft_model()     
    ref.requires_grad_(False)  
    ref.eval()

    trainable = [p for p in model.parameters() if p.requires_grad]
    total = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in trainable)
    print(f"设备 {device} / {dtype}，r={cfg['r']} alpha={cfg['alpha']} beta={BETA}，"
          f"可训参数 {trainable_num:,} ({trainable_num / total:.2%})")

    records = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    if LIMIT:
        records = records[:LIMIT]
    pairs = []
    for rec in records:
        s = encode(rec, tokenizer, MAX_LEN)
        if s:                   
            pairs.append(s)
    print(f"偏好对 {len(pairs)} 组")

    loader = DataLoader(
        pairs, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=lambda b: (collate(b, "chosen", tokenizer.pad_token_id),
                              collate(b, "rejected", tokenizer.pad_token_id)))
    opt = torch.optim.AdamW(trainable, lr=LR)   

    swanlab.init(
        project=SWANLAB_PROJECT,
        name=f"{SWANLAB_RUN}-{time.strftime('%m%d-%H%M%S')}",
        config={
            "base_model": MODEL_DIR.name,
            "sft": SFT_DIR.name,        
            "target_modules": cfg["target_modules"],
            "lora_r": cfg["r"],
            "lora_alpha": cfg["alpha"],
            "beta": BETA,
            "limit": LIMIT,
            "pairs": len(pairs),
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM,
            "batch_size_effective": BATCH_SIZE * GRAD_ACCUM,
            "steps_per_epoch": math.ceil(len(loader) / GRAD_ACCUM), 
            "lr": LR,
            "max_len": MAX_LEN,
            "trainable_params": sum(p.numel() for p in trainable),
            "device": device,
            "dtype": str(dtype),
        },
    )

    run = swanlab.get_run()
    run.define_metric("epoch/loss", x_axis="epoch/step", section_name="epoch")
    run.define_metric("epoch/margin", x_axis="epoch/step", section_name="epoch")
    run.define_metric("epoch/step", hidden=True, section_name="epoch") 

    model.train()
    global_step = 0             
    print("[训练中] 曲线见 swanlab")
    for epoch in range(EPOCHS):
        epoch_loss, epoch_margin, steps = 0.0, 0.0, 0
        opt.zero_grad()
        win_loss, win_margin, win_acc, win_n = 0.0, 0.0, 0.0, 0
        for i, (cb, rb) in enumerate(loader):
            cb = [t.to(device) for t in cb]   
            rb = [t.to(device) for t in rb]

            loss_c = seq_logp(model, *cb)
            loss_r = seq_logp(model, *rb)
            with torch.no_grad():             
                ref_c = seq_logp(ref, *cb)
                ref_r = seq_logp(ref, *rb)

            
            margin = (loss_c - ref_c) - (loss_r - ref_r)
            loss = -F.logsigmoid(BETA * margin).mean()
            (loss / GRAD_ACCUM).backward()                
            win_loss += loss.item()
            win_margin += margin.mean().item()
            win_acc += (margin > 0).float().mean().item()   
            win_n += 1

            if (i + 1) % GRAD_ACCUM == 0 or i + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)   
                opt.step()
                opt.zero_grad()                           
                global_step += 1
                swanlab.log({
                    "train/loss": win_loss / win_n,
                    "train/margin": win_margin / win_n,
                    "train/reward_accuracy": win_acc / win_n,
                }, step=global_step)
                epoch_loss += win_loss / win_n
                epoch_margin += win_margin / win_n
                steps += 1
                win_loss, win_margin, win_acc, win_n = 0.0, 0.0, 0.0, 0

        if steps:
            swanlab.log({
                "epoch/step": epoch + 1,
                "epoch/loss": epoch_loss / steps,      
                "epoch/margin": epoch_margin / steps,
            }, step=global_step)

    save_lora(model, CKPT_DIR, r=cfg["r"], alpha=cfg["alpha"],
              target_modules=cfg["target_modules"], base_model=MODEL_DIR.name,
              sft=str(SFT_DIR.name), samples=len(pairs), epochs=EPOCHS, beta=BETA)
    print(f"LoRA 权重已存到 {CKPT_DIR}")


if __name__ == "__main__":
    main()
