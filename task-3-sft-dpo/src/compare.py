import gc
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.chat import IM_END, IM_START, format_messages
from src.lora import inject_lora, load_lora

MODEL_DIR = ROOT / "models" / "Qwen2.5-0.5B"

MODELS = {
    "base": None,
    "SFT": ROOT / "ckpt" / "sft",
    "DPO": ROOT / "ckpt" / "dpo",
}

OUT_MD = ROOT / "comparison.md"
MAX_NEW_TOKENS = 128
REPETITION_PENALTY = 1.1   

PROMPTS = [
    "用一句话介绍你自己。",
    "上三角矩阵的特征值怎么求？",
    "把下面这句话翻译成英文：今天天气很好。",
    "什么是过拟合",
    "只要把模型参数调大就能解决所有问题吗？",
    "写C语言的helloworld",
]


def load_model(ckpt_dir, device, dtype):
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=dtype)
    cfg = None
    if ckpt_dir:
        cfg = json.loads((ckpt_dir / "config.json").read_text(encoding="utf-8"))
        inject_lora(model, cfg["target_modules"], r=cfg["r"], alpha=cfg["alpha"])
        load_lora(model, ckpt_dir)
    return model.to(device).eval(), cfg


def answer(model, tokenizer, prompt, device):
    """套 Qwen 模板，补 assistant 头"""
    text = format_messages([{"role": "user", "content": prompt}])
    text += f"{IM_START}assistant\n"
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                             repetition_penalty=REPETITION_PENALTY,
                             pad_token_id=tokenizer.eos_token_id)
    reply = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=False)
    for stop in (IM_END, "<|endoftext|>"):
        reply = reply.split(stop)[0]
    return reply.strip()


def main():
    sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认不是 utf-8
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    results = {p: {} for p in PROMPTS}   # prompt -> {标签: 回答}
    for label, ckpt_dir in MODELS.items():
        if ckpt_dir and not (ckpt_dir / "lora.pt").exists():
            print(f"[跳过] {label}：{ckpt_dir} 下没有 lora.pt")
            continue

        model, cfg = load_model(ckpt_dir, device, dtype)
        print(f"\n=== {label} ===") 

        for i, prompt in enumerate(PROMPTS, 1):
            reply = answer(model, tokenizer, prompt, device)
            results[prompt][label] = reply
            print(f"[{i}] {prompt}\n{reply}\n")

        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    lines = ["# base / SFT / DPO 对比"]
    for i, prompt in enumerate(PROMPTS, 1):
        lines += [f"## {i}. {prompt}", ""]
        for label in MODELS:                     
            if label not in results[prompt]:
                continue
            reply = results[prompt][label]
            lines += [f"**{label}**", "", reply or "（空回答）", ""]

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"已写入 {OUT_MD}")


if __name__ == "__main__":
    main()
