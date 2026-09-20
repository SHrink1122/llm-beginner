import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]   
sys.path.insert(0, str(ROOT))                

from src.attention import CausalSelfAttention, KVCache
from src.tokenizer import BPETokenizer


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.silu(self.linear1(x)))
    
class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, dropout=0.1):
        super().__init__()
        self.self_attn = CausalSelfAttention(d_model, num_heads, max_seq_len, rope_base=1e4)
        self.ffn = FeedForward(d_model, d_ff)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cache=None):
        # CausalAttention+Residual+Normalization
        attn_output = self.self_attn(self.norm1(x), cache)
        x = x + self.dropout(attn_output)

        # FFN+Residual+Normalization(layer)
        ff_output = self.ffn(self.norm2(x))
        x = x + self.dropout(ff_output)

        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, num_layers, 
                 num_heads=4, dropout=0.1, max_seq_len=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = max_seq_len
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, max_seq_len, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, vocab_size)

    def forward(self, ids, kv_cache=None, return_cache=False): 
        x = self.dropout(self.embedding(ids))

        if kv_cache is not None:
            caches = kv_cache
        elif return_cache:
            caches = [KVCache() for _ in self.blocks]   
        else:
            caches = [None] * len(self.blocks)
        for i, block in enumerate(self.blocks):
            x = block(x, caches[i])

        x = self.norm(x)
        logits = self.classifier(x)
        return (logits, caches) if return_cache else logits

    @staticmethod
    def apply_top_k(logits, top_k):
        if top_k is None or top_k <= 0:
            return logits
        # 防止 k 比词表还大
        k = min(top_k, logits.size(-1))

        threshold = torch.topk(logits, k, dim=-1).values[..., -1:]
        return logits.masked_fill(logits < threshold, float("-inf"))

    @staticmethod
    def apply_top_p(logits, top_p):
        if top_p is None or not (0.0 < top_p < 1.0):
            return logits

        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum_probs > top_p

        # 右移一位 至少留一个候选。
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False

        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))

        return logits.scatter(-1, sorted_indices, sorted_logits)

    def sample_next_token(self, logits, temperature=1.0, top_k=None, top_p=None):
        if temperature is None or temperature <= 0:
            # greedy
            return torch.argmax(logits, dim=-1, keepdim=True)

        # temperature 越大分布越平均，越小越尖锐（越接近 greedy）
        logits = logits / temperature
        logits = self.apply_top_k(logits, top_k)
        logits = self.apply_top_p(logits, top_p)

        probs = F.softmax(logits, dim=-1)

        return torch.multinomial(probs, num_samples=1)

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens, top_k=None, top_p=None, temperature=1.0):
        self.eval()
        device = next(self.parameters()).device

        # 对维度
        if isinstance(prompt_ids, torch.Tensor):
            ids = prompt_ids.to(device=device, dtype=torch.long)
        else:
            ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        cache = None

        # prompt 过长报错
        prompt_len = ids.size(1)
        assert prompt_len < self.block_size, (
            f"prompt 长度 {prompt_len} 已达到 block_size={self.block_size}，"
            f"无新 token 位置"
        )

        # prefill
        logits, cache = self(ids, kv_cache=cache, return_cache=True)
        logits = logits[:, -1, :]
        
        # decode
        for _ in range(max_new_tokens):
            if ids.size(1) >= self.block_size:
                break
            next_id = self.sample_next_token(
                logits, temperature=temperature, top_k=top_k, top_p=top_p
            )
            ids = torch.cat([ids, next_id], dim=1)
            logits, cache = self(ids[:, -1:], kv_cache=cache, return_cache=True)
            logits = logits[:, -1, :]

        return ids[0].tolist()
    

_ROOT = Path(__file__).resolve().parents[1]


def load_for_eval(ckpt_path, tokenizer_path=None):
    if tokenizer_path is None:
        tokenizer_path = _ROOT / "ckpt" / "tokenizer.json"
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # 加载模型
    model = MiniGPT(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])

    # 加载tokenizer
    tokenizer = BPETokenizer.from_pretrained(str(tokenizer_path))

    return model, tokenizer
