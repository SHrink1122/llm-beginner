# attn_fig.py
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from src.model import load_for_eval

# 中文字体：默认字体不含中文字形，坐标轴中文会显示为方块乱码，这里显式指定系统里的中文字体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans SC"]
plt.rcParams["axes.unicode_minus"] = False   # 避免负号显示成方块

LAYER = 5     
HEAD = 0       
CMAP = "viridis"


FIGSIZE = (9, 9)
TICK_FONTSIZE = 8
LABEL_FONTSIZE = 13
TITLE_FONTSIZE = 15


def save_heatmap(attn_w, labels, title, out_path):
    """画注意力热力图并保存。attn_w: (T, T)，labels: 每个 token 的字符标签"""
    n = len(labels)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    im = ax.imshow(attn_w, cmap=CMAP, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=TICK_FONTSIZE)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=TICK_FONTSIZE)
    ax.set_xlabel("key（被关注的词）", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("query（发起关注的词）", fontsize=LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("saved:", out_path)


if __name__ == "__main__":
    ckpt = Path(__file__).parent / "ckpt" / "best.pt"
    texts = {
        'pos': "轻便，方便携带，性能好，对出差人员来说非常不错",
        'neg': "房间太小。其他的都一般。",
        'long': "位置还可以，在市委市政府附近，要去商业区和步行街得打车，屋里有蚊子，很适合睡觉，但是会被该死的蚊子吵醒！卫生间挺大，但是设备很老旧。"
    }

    model, tokenize_fn = load_for_eval(str(ckpt))
    model.eval()

    out_dir = Path(__file__).parent / "figures"
    out_dir.mkdir(exist_ok=True)

    for key, text in texts.items():
        ids = tokenize_fn(text)
        n_chars = ids.size(0)                       

        with torch.no_grad():
            model(ids.unsqueeze(0))                 

        all_heads = model.blocks[LAYER].self_attn.attn_weights[0]
        labels = ["<cls>"] + list(text[:n_chars])

        # 第一个头
        save_heatmap(
            all_heads[HEAD].cpu().numpy(),
            labels,
            f"layer{LAYER} head{HEAD} — {key}",
            out_dir / f"attn_l{LAYER}_h{HEAD}_{key}.png",
        )
 