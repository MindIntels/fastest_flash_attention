"""
Fastest Flash Attention — Ops layer.

Provides the functional API with automatic CPU/GPU dispatch and
autograd support.
"""

from .attention_forward import fastest_flash_attn_forward
from .attention_backward import fastest_flash_attn_backward
from .attention_func import FastestFlashAttnFunc, fastest_flash_attn_func
from .fa4_func import FA4AttnFunc, fa4_attn_func, fa4_forward_dispatch
from .kv_cache import PagedKVCache, ContinuousKVCache
from .block_sparse import (
    create_causal_block_mask,
    create_sliding_window_block_mask,
    create_local_block_mask,
)

__all__ = [
    "fastest_flash_attn_forward",
    "fastest_flash_attn_backward",
    "FastestFlashAttnFunc",
    "fastest_flash_attn_func",
    "FA4AttnFunc",
    "fa4_attn_func",
    "fa4_forward_dispatch",
    "PagedKVCache",
    "ContinuousKVCache",
    "create_causal_block_mask",
    "create_sliding_window_block_mask",
    "create_local_block_mask",
]
