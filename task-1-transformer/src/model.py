# model.py 
import math
import torch
import torch.nn as nn
try:
    from block import TransformerBlock
    from tokenizer import CharTokenizer
except ImportError:
    from src.block import TransformerBlock
    from src.tokenizer import CharTokenizer


PAD_ID, UNK_ID, CLS_ID = 0, 1, 2


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, d_model, d_ff, num_layers, 
                 num_heads=4, dropout=0.1, max_len=128, pad_id=PAD_ID, cls_id=CLS_ID):
        super().__init__()
        # cls pooling
        self.pad_id, self.cls_id= pad_id, cls_id
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pe = PositionalEncoding(d_model, max_len + 1) # 算上cls位置
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, ids):
        batch_size, seq_length = ids.shape

        # 插入cls
        ids = torch.cat([ids.new_full((batch_size, 1), self.cls_id), ids], dim=1)
        # 生成padding_mask
        pad_mask = (ids == self.pad_id) # (batch_size, seq_len)
        attn_mask = pad_mask.unsqueeze(1).unsqueeze(2)

        x = self.embedding(ids)
        x = self.dropout(x)
        x = self.pe(x)
        for block in self.blocks:
            x = block(x, mask=attn_mask)
        x = x[:, 0,:] # pooling
        x = self.dropout(x)
        
        return self.classifier(x)

def load_for_eval(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # 加载模型
    model = TransformerClassifier(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])

    # 加载tokennizer
    tokenizer = CharTokenizer(ckpt["vocab"], max_len=ckpt["max_len"])

    return model, tokenizer.tokenize






        