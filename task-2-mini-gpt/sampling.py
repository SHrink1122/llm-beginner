import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.model import load_for_eval

import torch

CKPT = ROOT / "ckpt" / "best.pt"
MAX_NEW_TOKENS = 120
SEED = 0                    # 固定种子，让 samples.md 可复现
PROMPTS = ["春"]
OUT_MD = ROOT / "figures" / "samples.md"

# (标签, temperature, top_k, top_p)
STRATEGIES = [
    ("greedy",        0.0, None, None),
    ("T=0.5",         0.5, None, None),
    ("T=0.8",         0.8, None, None),
    ("T=1.2",         1.2, None, None),
    ("T=1.8",         1.8, None, None),
    ("top-k=5",       1.0, 5,    None),
    ("top-k=20",      1.0, 20,   None),
    ("top-p=0.9",     1.0, None, 0.9),
    ("top-p=0.95",    1.0, None, 0.95),
    ("T0.8+k20+p0.9", 0.8, 20,   0.9),
]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    model, tokenizer = load_for_eval(str(CKPT))

    lines = [
        "# mini-GPT 生成样例（不同采样策略对比）",
        "",
        f"`ckpt/best.pt`（{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M 参数，"
        f"vocab {tokenizer.vocab_size}，block_size {model.block_size}），"
        f"seed={SEED}，每条最多续写 {MAX_NEW_TOKENS} 个 token。",
        "",
    ]

    for prompt in PROMPTS:
        ids = tokenizer.encode(prompt)
        budget = min(MAX_NEW_TOKENS, model.block_size - len(ids))
        lines += ["---", "", f"## prompt：`{prompt}`", ""]

        for label, temperature, top_k, top_p in STRATEGIES:
            torch.manual_seed(SEED)          # 每条独立复位，策略之间互不干扰
            out = model.generate(ids, budget, top_k=top_k, top_p=top_p,
                                 temperature=temperature)
            text = tokenizer.decode(out[len(ids):]).strip()
            lines += [f"**{label}**", "", text or "（空）", ""]
            print(f"\n[{prompt}] {label}\n{text}")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n已写入 {OUT_MD}")


if __name__ == "__main__":
    main()
