import json
from pathlib import Path

import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None: # 防止nn.linear 用 bias=False 构造
            self.base.bias.requires_grad_(False)
        self.scaling = alpha / r
        # qwen采用bf16 torch默认fp32会报错
        self.lora_A = nn.Linear(base.in_features, r, bias=False,
                                dtype=base.weight.dtype, device=base.weight.device)
        self.lora_B = nn.Linear(r, base.out_features, bias=False,
                                dtype=base.weight.dtype, device=base.weight.device)
        nn.init.zeros_(self.lora_B.weight)   # B 初始置零

    def forward(self, x):
        return self.base(x) + self.scaling * self.lora_B(self.lora_A(x))


def inject_lora(model, target_modules, r=8, alpha=16):
    model.requires_grad_(False)
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and name.endswith(tuple(target_modules)):
            _set_submodule(model, name, LoRALinear(module, r, alpha))
    return model


def merge_lora(model):
    # 推理前merge
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            w = module.base.weight
            delta = module.scaling * (module.lora_B.weight @ module.lora_A.weight)  
            w.data += delta.to(w.dtype)
            _set_submodule(model, name, module.base)
    return model


def lora_state_dict(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if ".lora_" in k}


def save_lora(model, ckpt_dir, **config):
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(lora_state_dict(model), ckpt_dir / "lora.pt")
    # r，alpha， target_modules 同样记录
    (ckpt_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return ckpt_dir


def load_lora(model, ckpt_dir):
    state = torch.load(Path(ckpt_dir) / "lora.pt", map_location="cpu")
    model.load_state_dict(state, strict=False)
    return model


def _set_submodule(model, name, new_module):
    # 替换模块
    parent_path, _, attr = name.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    setattr(parent, attr, new_module)
