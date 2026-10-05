from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    # T3d SWA 池（pool 1）专用；单池形态恒 None（零开销路径）
    slot_mapping_swa: torch.Tensor | None = None
    context_lens_swa: torch.Tensor | None = None    # 窗口内可见长度 = L - window_start
    block_tables_swa: torch.Tensor | None = None    # 窗口裁剪后的块表

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0,
                slot_mapping=None, context_lens=None, block_tables=None,
                slot_mapping_swa=None, context_lens_swa=None, block_tables_swa=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                       slot_mapping, context_lens, block_tables,
                       slot_mapping_swa, context_lens_swa, block_tables_swa)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
