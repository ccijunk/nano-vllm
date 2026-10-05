import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class SidecarConfig:
    """C1 侧车 config.json → HF 兼容属性名。

    ModelRunner 与 qwen3 式模型代码按 HF 命名消费（num_hidden_layers 等），零改动适配；
    字段名映射 = NanoConfig（snake 短名）→ HF 风格（doc/contracts/weight_format.md）。
    """

    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool
    dtype: torch.dtype                 # "bfloat16" → torch.bfloat16（C1 dtype 字段）
    hidden_act: str = "silu"
    # T3d 多池声明（SWA 层自报，C3 pools=2 的消费协议，doc/topics/t3d_swa_multipool.md §1）：
    # 两字段齐备且列表非空 ⟺ 两池形态；缺省 ⟺ 单池（现状）
    sliding_window: int | None = None
    sliding_window_layers: tuple[int, ...] | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> "SidecarConfig":
        raw = json.loads(Path(path).read_text())
        swa_layers = raw.get("sliding_window_layers")
        return cls(
            vocab_size=raw["vocab_size"],
            hidden_size=raw["hidden_size"],
            num_hidden_layers=raw["n_layers"],
            num_attention_heads=raw["n_heads"],
            num_key_value_heads=raw["n_kv_heads"],
            head_dim=raw["head_dim"],
            intermediate_size=raw["intermediate_size"],
            rms_norm_eps=raw["rms_norm_eps"],
            rope_theta=raw["rope_theta"],
            max_position_embeddings=raw["max_position_embeddings"],
            tie_word_embeddings=raw["tie_word_embeddings"],
            dtype=getattr(torch, raw["dtype"]),
            sliding_window=raw.get("sliding_window"),
            sliding_window_layers=tuple(swa_layers) if swa_layers else None,
        )


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: SidecarConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1        # 全池（pool 0）块数，ModelRunner.allocate_kv_cache 填
    num_kvcache_blocks_swa: int = -1    # SWA 池（pool 1）块数；-1 = 单池形态

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = SidecarConfig.from_json(os.path.join(self.model, "config.json"))
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
