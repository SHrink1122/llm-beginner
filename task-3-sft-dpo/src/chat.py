from functools import lru_cache
from pathlib import Path
from transformers import AutoTokenizer
import torch

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "Qwen2.5-0.5B"
DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"   


@lru_cache(maxsize=1) # 不重复找
def _tokenizer():
    return AutoTokenizer.from_pretrained(str(MODEL_DIR))


def format_messages(messages):
    # qwen template 简化实现
    if not messages or messages[0]["role"] != "system":
        messages = [{"role": "system", "content": DEFAULT_SYSTEM}] + list(messages)

    return "".join(f"{IM_START}{m['role']}\n{m['content']}{IM_END}\n" for m in messages)


def build_labels(input_ids, messages):
    labels = torch.as_tensor(input_ids).clone()
    labels[:] = -100  
    text = format_messages(messages)

    encoded = _tokenizer()(text, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]  
    ids = encoded["input_ids"]

    pos = 0  
    for m in messages:
        if m["role"] != "assistant":
            continue  # user / system 不训练
        
        # 找到要替换的assitant的content索引[start, end)
        start = text.find(m["content"], pos)  # pos 开始 避免有重复content只找那一个
        if start < 0:
            continue

        end = start + len(m["content"]) + len(IM_END)
        pos = end

        # 训练content 换tokenid
        for i, (a, b) in enumerate(offsets):
            if i < len(labels) and a >= start and 0 < b <= end:
                labels[i] = ids[i]

    return labels
