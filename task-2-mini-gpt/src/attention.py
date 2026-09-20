import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]   
sys.path.insert(0, str(ROOT))                

from src.rope import apply_rotary_pos_emb, precompute_rope_cos_sin   

class KVCache:
    def __init__(self):
        self.k = None
        self.v = None

    @property
    def length(self):
        return 0 if self.k is None else self.k.shape[2]

    def update(self, k, v):
        if self.k is None:
            self.k, self.v = k, v
        else:
            self.k = torch.cat([self.k, k], dim=2)
            self.v = torch.cat([self.v, v], dim=2)

    def reset(self):
        self.k = None
        self.v = None


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len, rope_base=1e4):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model // self.num_heads 
        self.max_seq_len = max_seq_len

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # 算困惑度时窗口是 block_size + 1 个token
        cos, sin = precompute_rope_cos_sin(self.d_k, max_seq_len + 1, rope_base=rope_base)
        self.register_buffer("cos", cos, persistent=False)  # (max_seq_len, d_k)
        self.register_buffer("sin", sin, persistent=False)
        
    def split_heads(self, x):
        # (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
        batch_size, seq_len = x.shape[:2]
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def concat_heads(self, x):
        # (batch, num_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        batch_size, _, seq_len, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, x, cache=None):
        start_pos = 0
        if cache is not None:
            start_pos = cache.length  
        
        seq_len = x.shape[1]
        # 线性变换+分头
        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x)) #(batch, num_heads, seq_len, d_k)
        
        if start_pos + seq_len > self.cos.size(0):
            raise ValueError(
                f"start_pos超出范围：需要 {start_pos + seq_len} 个位置，"
            )

        cos = self.cos[start_pos: start_pos + seq_len]
        sin = self.sin[start_pos: start_pos + seq_len]
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        if cache is not None:
            cache.update(K, V)
            K, V = cache.k, cache.v

        causal_mask = torch.triu(torch.ones(Q.shape[2], K.shape[2], device=Q.device), 
                                 diagonal=1 + start_pos).bool()

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        scores = scores.masked_fill(causal_mask, float("-inf")) 
        attn = F.softmax(scores, dim=-1)

        return self.W_o(self.concat_heads(attn @ V))


        

        
