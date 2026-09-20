import json
import math
import sys
import time
import swanlab
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))          

from src.model import MiniGPT
from src.tokenizer import BPETokenizer
from src.cache import encode_cached


# ---------------- 超参数 ----------------
D_MODEL = 96        
N_HEADS = 4             
N_LAYERS = 4           
D_FF = 4 * D_MODEL      
DROPOUT = 0.3          
BLOCK_SIZE = 128       

BATCH_SIZE = 32
LR = 3e-4
EPOCHS = 500             
WARMUP_RATIO = 0.05
COSINE_POWER = 1.0      # 余弦退火陡峭度
GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.3     

EVAL_EVERY = 5          # 每几个 epoch 算一次 dev ppl
PATIENCE = 6         # 早停
EVAL_TOKENS = 4096      # 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "ckpt"
TOKENIZER_PATH = CKPT_DIR / "tokenizer.json"
CKPT_PATH = CKPT_DIR / "best.pt"
LOG_PATH = CKPT_DIR / "train_log.json"   # 训练历史，画曲线用
FIG_DIR = ROOT / "figures"
FIG_PATH = FIG_DIR / "ppl_curve.png"
CACHE_DIR = DATA_DIR / "cache"           # encode 结果的缓存，词表/语料变了会自动重算 

SWANLAB_PROJECT = "mini-gpt"  
MISSION = "poetry"

class LMDataset(Dataset):
    def __init__(self, ids, block_size):
        self.ids = ids
        self.block_size = block_size
        # 丢最后不足block_size的token
        self.num_blocks = max(0, (len(ids) - 1) // block_size)

    def __len__(self):
        return self.num_blocks

    def __getitem__(self, index):
        start = index * self.block_size
        chunk = self.ids[start: start + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


@torch.no_grad()
def evaluate_ppl(model, ids, block_size, max_tokens=EVAL_TOKENS):
    model.eval()
    ids = ids[:max_tokens]
    total_nll, total_tok = 0.0, 0
    for start in range(0, max(1, len(ids) - 1), block_size):
        window = ids[start: start + block_size + 1]
        if len(window) < 2: # 如果最后一块只剩1个token，跳出
            break
        chunk = torch.tensor([window], dtype=torch.long, device=DEVICE)
        logits = model(chunk)
        total_nll += F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)), # (1, L, V) → (L-1, V)
            chunk[:, 1:].reshape(-1),
            reduction="sum",
        ).item()
        total_tok += chunk.size(1) - 1
    model.train()
    return math.exp(total_nll / total_tok) if total_tok else float("inf")

def load_threshold():
    info_path = DATA_DIR / "dataset_info.json"
    if not info_path.exists():
        return None
    try:
        return float(json.loads(info_path.read_text(encoding="utf-8")).get("ppl_threshold"))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def read_best_ppl(path):
    """读盘上留着的 best.pt 里记的 dev_ppl。

    """
    if not path.exists():
        return float("inf")
    try:
        return float(torch.load(path, map_location="cpu")["dev_ppl"])
    except Exception as e:
        return float("inf")


def plot_curve(history, threshold=None):
    """画困惑度曲线"""
    import matplotlib
    matplotlib.use("Agg")           #
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    surface, ink, ink2, muted = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
    grid, axis = "#e1e0d9", "#c3c2b7"
    train_color, dev_color = "#2a78d6", "#eb6834"

    pts = [h for h in history if h["dev_ppl"] is not None]
    epochs = [h["epoch"] for h in pts]
    train_ppl = [h["train_ppl"] for h in pts]
    dev_ppl = [h["dev_ppl"] for h in pts]

    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)

    for ys, color, label in ((train_ppl, train_color, "train"),
                             (dev_ppl, dev_color, "dev")):
        ax.plot(epochs, ys, color=color, lw=2, solid_capstyle="round",
                marker="o", ms=6, markeredgecolor=surface, markeredgewidth=1.5,
                label=label, zorder=3)


    lower, upper = sorted([(train_ppl[-1], "train"), (dev_ppl[-1], "dev")])
    for value, name in (lower, upper):
        dy = 8 if name == upper[1] else -16    
        ax.annotate(f"{name} {value:.1f}", xy=(epochs[-1], value),
                    xytext=(6, dy), textcoords="offset points",
                    color=ink2, fontsize=9, va="center", zorder=4)

    if threshold:
        ax.axhline(threshold, color=muted, lw=1.5, ls="--", zorder=2)
        ax.text(0.995, threshold, f"threshold {threshold:g}", color=ink2, fontsize=9,
                ha="right", va="bottom",
                transform=ax.get_yaxis_transform(), zorder=4)


    ax.set_yscale("log")
    lo = min(min(train_ppl), min(dev_ppl), threshold or float("inf"))
    hi = max(max(train_ppl), max(dev_ppl), threshold or 0)
    ax.set_ylim(lo / 1.2, hi * 1.25)
    ticks = [t for t in (1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000, 2000, 3000, 5000)
             if lo / 1.2 <= t <= hi * 1.25]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:,}" for t in ticks])
    ax.minorticks_off()

    ax.set_title("mini-GPT 预训练困惑度曲线", color=ink, fontsize=13, pad=12, loc="left")
    ax.set_xlabel("epoch", color=muted, fontsize=10)
    ax.set_ylabel("困惑度", color=muted, fontsize=10)
    ax.grid(axis="y", color=grid, lw=1, ls="-", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(axis)
    ax.tick_params(colors=muted, labelsize=9, length=0)
    ax.legend(loc="upper right", frameon=False, fontsize=10, labelcolor=ink2)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150, facecolor=surface, bbox_inches="tight")
    plt.close(fig)
    print(f"困惑度曲线已保存到 {FIG_PATH}")
    return FIG_PATH


def train():
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- 准备数据 ----------
    train_text = (DATA_DIR / "train.txt").read_text(encoding="utf-8")
    dev_text = (DATA_DIR / "dev.txt").read_text(encoding="utf-8")

    if TOKENIZER_PATH.exists():
        tokenizer = BPETokenizer.from_pretrained(str(TOKENIZER_PATH))
    else:
        print(f"{TOKENIZER_PATH} 不存在，训一个词表")
        tokenizer = BPETokenizer()
        tokenizer.train(train_text, vocab_size=347)
        tokenizer.save(str(TOKENIZER_PATH))

    train_ids = encode_cached(tokenizer, train_text, CACHE_DIR / "train_ids.npz")
    dev_ids = encode_cached(tokenizer, dev_text, CACHE_DIR / "dev_ids.npz")

    train_ds = LMDataset(train_ids, BLOCK_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    config = dict(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        d_ff=D_FF,
        num_layers=N_LAYERS,
        num_heads=N_HEADS,
        dropout=DROPOUT,
        max_seq_len=BLOCK_SIZE,
    )
    model = MiniGPT(**config).to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量 {num_params / 1e6:.2f}M | device={DEVICE}")

    best_on_disk = read_best_ppl(CKPT_PATH)

    # ---------- SwanLab 记录 ----------
    swanlab.init(
        project=SWANLAB_PROJECT,
        name = f"mini-gpt-{MISSION}-{time.strftime('%m%d-%H%M%S')}",
        config={
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
            "n_layers": N_LAYERS,
            "d_ff": D_FF,
            "dropout": DROPOUT,
            "block_size": BLOCK_SIZE,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "epochs": EPOCHS,
            "warmup_ratio": WARMUP_RATIO,
            "grad_clip": GRAD_CLIP,
            "weight_decay": WEIGHT_DECAY,
            "eval_every": EVAL_EVERY,
            "patience": PATIENCE,
            "vocab_size": tokenizer.vocab_size,
            "num_params": num_params,
            "device": DEVICE,
            "train_tokens": len(train_ids),
            "dev_tokens": len(dev_ids),
        },
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # warmup, 余弦退火
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = max(1, int(WARMUP_RATIO * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps            # 线性升温
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress ** COSINE_POWER))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    print(f" {len(train_loader)} batch/epoch × {EPOCHS} epoch = {total_steps} 步")
    
    # ---------- 训练循环 ----------
    best_dev_ppl = float("inf")     
    bad_evals = 0
    global_step = 0
    history = []            # 画曲线用
    print(f"[训练中] 曲线见swanlab")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss, epoch_tok = 0.0, 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            global_step += 1
            epoch_loss += loss.item() * y.numel()
            epoch_tok += y.numel()
            swanlab.log({"train/loss": loss.item()}, step=global_step)

        train_ppl = math.exp(epoch_loss / epoch_tok)
        dev_ppl = None

        if epoch % EVAL_EVERY == 0 or epoch == EPOCHS:
            dev_ppl = evaluate_ppl(model, dev_ids, BLOCK_SIZE)
            history.append({"epoch": epoch, "step": global_step,
                            "train_ppl": round(train_ppl, 3), "dev_ppl": round(dev_ppl, 3)})
            LOG_PATH.write_text(json.dumps(history, ensure_ascii=False, indent=2),
                                encoding="utf-8")
            if dev_ppl < best_dev_ppl:
                best_dev_ppl = dev_ppl
                bad_evals = 0
                if dev_ppl < best_on_disk:
                    best_on_disk = dev_ppl
                    torch.save({
                        "state_dict": model.state_dict(),
                        "config": config,          # load_for_eval 靠它重建模型
                        "dev_ppl": best_dev_ppl,
                        "epoch": epoch,
                        "step": global_step,
                    }, CKPT_PATH)
                    print(f"    -> 新的最优 dev_ppl={dev_ppl:.2f}，已保存到 {CKPT_PATH}")
            else:
                bad_evals += 1

        lr_now = optimizer.param_groups[0]["lr"]
        metrics = {"train/ppl": train_ppl, "train/lr": lr_now}
        if dev_ppl is not None:
            metrics["dev/ppl"] = dev_ppl
            metrics["dev/best_ppl"] = best_dev_ppl
        swanlab.log(metrics, step=global_step)

        # dev_ppl 连续都不降提早停止
        if bad_evals >= PATIENCE:
            print(f"[早停] dev_ppl 连续 {PATIENCE} 次评估没有变好，第 {epoch} 个 epoch 停")
            break


    threshold = load_threshold()

    print(f"[完成] 本次 run 最佳 dev_ppl={best_dev_ppl:.2f}")
    if best_dev_ppl > best_on_disk:
        print(f"未更新ckpt")
    print(f"目前最优 dev_ppl={best_on_disk:.2f} | checkpoint: {CKPT_PATH}")

    fig_path = plot_curve(history, threshold)

    # 本地困惑度曲线上传 SwanLab
    if fig_path:
        swanlab.log({"ppl_curve": swanlab.Image(str(fig_path), caption="mini-GPT 预训练困惑度曲线")},
                    step=global_step)

if __name__ == "__main__":
    train()
