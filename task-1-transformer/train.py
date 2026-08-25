# train.py
import torch
import pandas as pd
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
import math

from src.tokenizer import TextDataset, build_tokenizer, collate_fn
from src.model import TransformerClassifier


# 超参数
D_MODEL = 256
N_HEADS = 8
N_LAYERS = 6
D_FF = 4 * D_MODEL
DROPOUT = 0.1
MAX_SEQ_LEN = 256

BATCH_SIZE = 64
WARM_UP_RATIO = 0.1
LR = 3e-4
EPOCHS = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = Path(__file__).resolve().parent
CKPT_DIR = ROOT / "ckpt"
CKPT_PATH = CKPT_DIR / "best.pt"

# 特殊token的id
PAD_ID, UNK_ID, CLS_ID = 0, 1, 2

def save_checkpoint(model, tokenizer, config, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config,
            "vocab": sorted(tokenizer.char2id.keys()),
            "max_len": tokenizer.max_len,
        },
        str(path),
    )

def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for ids, labels in loader:
            ids = ids.to(DEVICE)
            logits = model(ids)
            pred = logits.argmax(dim=-1)
            correct += (pred == labels.to(DEVICE)).sum().item()
            total += labels.numel()
    return correct / total

def train():
    train_df = pd.read_parquet(ROOT / "data" / "train.parquet")
    val_df = pd.read_parquet(ROOT / "data" / "validation.parquet")

    num_classes = 2
    
    tokenizer = build_tokenizer(train_df["text"].tolist(), MAX_SEQ_LEN)

    train_loader = DataLoader(
        TextDataset(train_df["text"], train_df["label"], tokenizer),
        batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0
    )
    vali_loader = DataLoader(
        TextDataset(val_df["text"], val_df["label"], tokenizer),
        batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    config = dict(
        vocab_size=tokenizer.vocab_size,
        num_classes=num_classes,
        d_model=D_MODEL,
        d_ff=D_FF,
        num_layers=N_LAYERS,
        num_heads=N_HEADS,
        dropout=DROPOUT,
        max_len=MAX_SEQ_LEN,
        pad_id=PAD_ID,
        cls_id=CLS_ID,
    )
    model = TransformerClassifier(**config).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(),lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(WARM_UP_RATIO * total_steps)
    print(f"total steps: {total_steps}")

    # 余弦退火 + warm up
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)          # 线性升温
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))  # 余弦衰减到 0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_acc = -1.0
    print(f"[训练] device={DEVICE} | 样本数 train={len(train_df)} val={len(val_df)}")
    print(f'总batch数={len(train_loader)}')

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch_idx, (ids, labels) in enumerate(train_loader):
            ids, labels = ids.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(ids)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * ids.size(0)
            if batch_idx % 50 == 0:  # 每50个batch打印一次
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | loss: {loss.item():.4f}")

        val_acc = evaluate(model, vali_loader)
        avg_loss = total_loss / len(train_df)
        print(f"epoch {epoch + 1}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

        # early stopping by val acc
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, tokenizer, config, CKPT_PATH)
            print(f"    -> 保存新最优模型到 {CKPT_PATH}")

    print(f"[完成] 最佳 val_acc={best_acc:.4f}，checkpoint 在 {CKPT_PATH}")
       
if __name__ == "__main__":
    train()