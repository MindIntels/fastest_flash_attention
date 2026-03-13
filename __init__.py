"""
Fastest Flash Attention — Production-ready flash attention combining
Dao-AILab/flash-attention and xlite-dev/ffpa-attn techniques.

Key Features
============
- **Fine-Grained Pipelining (FFPA)**: Overlaps data loading with computation
  at warp/register level for maximum throughput.
- **Online Softmax with Deferred Rescaling**: Single normalization per Q-block,
  minimising non-matmul FLOPs.
- **Mixed-Precision**: FP16/BF16 compute with FP32 accumulation; optional FP8.
- **Block-Sparse Attention**: Skip entire tiles via configurable block masks.
- **Causal & Sliding-Window**: Native causal masking with block-level early-exit
  and sliding-window support.
- **Paged KV-Cache**: Memory-efficient autoregressive decoding with page tables.
- **Softcap (Gemma-2 style)**: Tanh-based logit capping for training stability.
- **GQA / MQA**: Grouped-query and multi-query attention support.
- **Triton GPU Kernels**: Auto-dispatch to Triton kernels when available,
  fallback to optimised CPU reference.
- **Autograd Support**: Custom backward pass with activation recomputation.

Public API
==========
Functional:
    fastest_flash_attn_forward   — Forward pass (auto-dispatch CPU/GPU).
    fastest_flash_attn_backward  — Backward pass (gradient computation).
    fastest_flash_attn_func      — Combined fwd+bwd with autograd.

Modules:
    FastestFlashMHA   — Multi-Head Attention module.
    FastestFlashGQA   — Grouped-Query Attention module.
    FastestFlashMLA   — Multi-Latent Attention module (DeepSeek-style).

Utilities:
    PagedKVCache       — Paged KV-cache manager.
    AttentionConfig    — Configuration dataclass.
    auto_select_block_size — Adaptive block-size heuristic.
"""

from .config import AttentionConfig
from .utils import auto_select_block_size, check_triton_available

from .ops.attention_forward import fastest_flash_attn_forward
from .ops.attention_backward import fastest_flash_attn_backward
from .ops.attention_func import FastestFlashAttnFunc, fastest_flash_attn_func
from .ops.fa4_func import FA4AttnFunc, fa4_attn_func, fa4_forward_dispatch
from .ops.kv_cache import PagedKVCache, ContinuousKVCache
from .ops.block_sparse import (
    create_causal_block_mask,
    create_sliding_window_block_mask,
    create_local_block_mask,
)

from .modules.flash_mha import FastestFlashMHA
from .modules.flash_gqa import FastestFlashGQA
from .modules.flash_mla import FastestFlashMLA
from .modules.flash_fa4 import FastestFlashFA4

__version__ = "1.0.0"

__all__ = [
    # Config
    "AttentionConfig",
    # Functional
    "fastest_flash_attn_forward",
    "fastest_flash_attn_backward",
    "fastest_flash_attn_func",
    "FastestFlashAttnFunc",
    # FA4 Functional
    "fa4_attn_func",
    "fa4_forward_dispatch",
    "FA4AttnFunc",
    # Modules
    "FastestFlashMHA",
    "FastestFlashGQA",
    "FastestFlashMLA",
    "FastestFlashFA4",
    # KV-Cache
    "PagedKVCache",
    "ContinuousKVCache",
    # Block Sparse
    "create_causal_block_mask",
    "create_sliding_window_block_mask",
    "create_local_block_mask",
    # Utilities
    "auto_select_block_size",
    "check_triton_available",
]
