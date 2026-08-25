# attenion.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def scaled_dot_product_attention(Q, K, V, mask=None):
    #q.shape=(batch, num_heads, seq_len, d_k)
    d_k = Q.shape[-1]
    #计算注意力分数
    scores = Q @ (K.transpose(-2, -1)) / math.sqrt(d_k)
    #掩码机制
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9) 

    attn_weights = F.softmax(scores, dim=-1)
    return attn_weights @ V

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model // self.num_heads 

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        batch_size, seq_len = x.shape[:2]
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def concat_heads(self, x):
        batch_size, _, seq_len, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, x, mask=None):
        # 线性变换+分头
        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))

        # 注意力分数计算
        attn_output = scaled_dot_product_attention(Q, K, V, mask)

        output = self.W_o(self.concat_heads(attn_output))

        return output


if __name__ == "__main__":
    # 自测：直接跑 attention.py 时才执行；被 import 时不会打印
    mha = MultiHeadAttention(d_model=128, num_heads=4)
    x = torch.randn(2, 10, 128)          # (B=2, T=10)
    mask = torch.zeros(2, 1, 10, 10).bool()  # 全 False，不屏蔽
    out = mha(x, mask)
    print(out.shape)  # 应为 (2, 10, 128)