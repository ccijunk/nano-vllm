"""T3b · NanoForCausalLM —— C1 键名的引擎侧实现（models/qwen3.py 同构）。

C1 键名 = HF Qwen3 风格（T1a 有意设计）→ packed_modules_mapping 与 Qwen3ForCausalLM 一致：
self_attn.q_proj/k_proj/v_proj → qkv_proj 分片、mlp.gate_proj/up_proj → gate_up_proj 分片，
utils/loader.py 的 replace + weight_loader 逻辑原样可用。
上游 models/qwen3.py 保留不改（对照物）；本文件 = engine 主路径（T3b）。

KV 组织声明（C3 同构，engine 侧复制——跨仓库无包依赖，数值一致性由 tests 锚定，
doc/topics/t3_nano_vllm.md §6.4）：GQA 单池，per-token KV = 2(K+V) × n_kv × head_dim × 2B(bf16)。
"""

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
import torch.distributed as dist

from nanovllm.config import SidecarConfig
from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


@dataclass(frozen=True, eq=False)
class KVOrg:
    """C3 KVOrganization 的 engine 侧同构声明（pools / per_token_bytes / layout）。"""

    pools: int
    per_token_bytes: Callable[[SidecarConfig], int]
    layout: str


class NanoAttention(nn.Module):

    def __init__(self, config: SidecarConfig, pool_idx: int = 0, window: int = 0) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
        )
        self.attn = Attention(self.num_heads, self.head_dim, self.scaling, self.num_kv_heads,
                              pool_idx=pool_idx, window=window)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        return self.o_proj(o.flatten(1, -1))


class NanoMLP(nn.Module):

    def __init__(self, config: SidecarConfig) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size, [config.intermediate_size] * 2,
        )
        self.down_proj = RowParallelLinear(config.intermediate_size, config.hidden_size)
        assert config.hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class NanoDecoderLayer(nn.Module):

    def __init__(self, config: SidecarConfig, pool_idx: int = 0, window: int = 0) -> None:
        super().__init__()
        self.self_attn = NanoAttention(config, pool_idx=pool_idx, window=window)
        self.mlp = NanoMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class NanoModel(nn.Module):

    def __init__(self, config: SidecarConfig) -> None:
        super().__init__()
        # T3d 层→池映射（design §1）：sliding_window_layers 中的层 = pool 1（SWA），其余 = pool 0
        swa_layers = set(config.sliding_window_layers or ())
        window = config.sliding_window or 0
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            NanoDecoderLayer(config, pool_idx=1 if i in swa_layers else 0, window=window if i in swa_layers else 0)
            for i in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class NanoForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    kv_org = KVOrg(
        pools=1,
        per_token_bytes=lambda cfg: 2 * cfg.num_key_value_heads * cfg.head_dim * cfg.dtype.itemsize,
        layout="paged_kt_vt",
    )

    def __init__(self, config: SidecarConfig) -> None:
        super().__init__()
        # T3d（design §1）：层→池映射由侧车 config 决定 kv_org 声明——有 SWA 层 = 双池
        swa_layers = config.sliding_window_layers or ()
        if swa_layers:
            assert config.sliding_window and config.sliding_window > 0, "sliding_window_layers 需配 sliding_window"
            assert max(swa_layers) < config.num_hidden_layers, f"SWA 层号越界: {max(swa_layers)}"
            self.kv_org = KVOrg(
                pools=2,
                per_token_bytes=lambda cfg: 2 * cfg.num_key_value_heads * cfg.head_dim * cfg.dtype.itemsize,
                layout="paged_kt_vt|paged_kt_vt_swa",
            )
        else:
            self.kv_org = type(self).kv_org  # 单池默认（类属性）
        self.model = NanoModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
