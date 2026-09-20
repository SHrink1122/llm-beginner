import torch

def precompute_rope_cos_sin(dim, max_len, rope_base=1e4):

    # 每个维度对的旋转频率 theta_i = rope_base^(-2i/dim)，共 dim/2 个
    inv_freq = 1.0 / (rope_base ** (torch.arange(0, dim, 2).float() / dim))

    t = torch.arange(max_len)
    # (max_len, dim/2)，freqs[m][i] = m * theta_i
    freqs = torch.outer(t, inv_freq)

   
    cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return cos, sin


def apply_rotary_pos_emb(q, k, cos, sin):
    def rotate_half(x):
        # 把后半段取负放到前半段：x = [x1, x2] -> [-x2, x1]
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed
