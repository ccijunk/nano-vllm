"""T3b · E-B2 参考实现：fp32 朴素 GQA 前向（无 flash / 分页 / TP，逐算子）。

数值约定与引擎逐条对齐（doc/topics/t3_nano_vllm.md §4 E-B2）：
- RMSNorm：fp32 内 rsqrt(mean(x²)+eps) · w（nanovllm/layers/layernorm.py 同构）
- RoPE：对半旋转（chunk 对半，cos/sin 各 D/2，cache 布局 cat(cos,sin)），q/k 先 norm 后 rope
- GQA：q head i → kv head i // (n_heads // n_kv_heads)（flash_attn 的 kv 映射同款）
- tie_word_embeddings：logits = h @ embed_tokens.weight.T（lm_head 不存，C1 §命名约定）
"""

import torch

# C1 基线超参（doc/contracts/weight_format.md 键表注记；单一事实源，conftest 复用）
CFG = dict(
    vocab_size=42,
    hidden_size=256,
    n_layers=2,
    n_heads=8,
    n_kv_heads=2,
    head_dim=32,
    intermediate_size=768,
    rms_norm_eps=1e-6,
    rope_theta=10000.0,
    max_position_embeddings=1024,
    tie_word_embeddings=True,
    dtype="bfloat16",
    attn_impl="naive",
)


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rope(x: torch.Tensor, theta: float) -> torch.Tensor:
    # x [T, H, D]；teacher-forced 全序列 positions = arange(T)（引擎 prepare_prefill 同款）
    T, _, D = x.shape
    inv_freq = 1.0 / (theta ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
    freqs = torch.outer(torch.arange(T, dtype=torch.float32), inv_freq)   # [T, D/2]
    cos, sin = freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)         # [T, 1, D/2]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


@torch.no_grad()
def forward(weights: dict[str, torch.Tensor], ids: list[int], cfg: dict) -> torch.Tensor:
    """返回末位 logits（fp32，[vocab_size]）。weights = C1 fp32 master，cfg = C1 侧车字段。"""
    W = weights
    L, H, KVH, D = cfg["n_layers"], cfg["n_heads"], cfg["n_kv_heads"], cfg["head_dim"]
    eps, theta = cfg["rms_norm_eps"], cfg["rope_theta"]
    group = H // KVH
    scale = D ** -0.5
    T = len(ids)
    x = W["model.embed_tokens.weight"][torch.tensor(ids)].float()   # [T, hidden]
    residual = x
    for i in range(L):
        p = f"model.layers.{i}."
        if i == 0:   # 引擎首层 residual=None：rms_forward 分支
            h = rms_norm(x, W[p + "input_layernorm.weight"].float(), eps)
        else:        # 引擎 add_rms_forward：residual = x + residual，再 norm
            residual = residual + x
            h = rms_norm(residual, W[p + "input_layernorm.weight"].float(), eps)
        q = (h @ W[p + "self_attn.q_proj.weight"].float().T).view(T, H, D)
        k = (h @ W[p + "self_attn.k_proj.weight"].float().T).view(T, KVH, D)
        v = (h @ W[p + "self_attn.v_proj.weight"].float().T).view(T, KVH, D)
        q = rms_norm(q, W[p + "self_attn.q_norm.weight"].float(), eps)
        k = rms_norm(k, W[p + "self_attn.k_norm.weight"].float(), eps)
        q, k = rope(q, theta), rope(k, theta)
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
        scores = torch.einsum("thd,shd->hts", q, k) * scale          # [H, T, T]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        o = torch.softmax(scores.masked_fill(mask, float("-inf")), -1) @ v.transpose(0, 1)
        x = o.transpose(0, 1).reshape(T, H * D) @ W[p + "self_attn.o_proj.weight"].float().T
        residual = residual + x
        h = rms_norm(residual, W[p + "post_attention_layernorm.weight"].float(), eps)
        gate = h @ W[p + "mlp.gate_proj.weight"].float().T
        up = h @ W[p + "mlp.up_proj.weight"].float().T
        x = (torch.nn.functional.silu(gate) * up) @ W[p + "mlp.down_proj.weight"].float().T
    out = rms_norm(x + residual, W["model.norm.weight"].float(), eps)
    return out[-1] @ W["model.embed_tokens.weight"].float().T
