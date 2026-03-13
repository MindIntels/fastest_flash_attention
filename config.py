"""
Attention configuration — centralised knobs for Fastest Flash Attention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch


@dataclass
class AttentionConfig:
    """All tuneable parameters for the Fastest Flash Attention kernel.

    Attributes
    ----------
    block_size_q : int
        Tile size along the query (row) axis.  ``0`` → auto-select.
    block_size_kv : int
        Tile size along the key/value (column) axis.  ``0`` → auto-select.
    causal : bool
        Apply lower-triangular causal mask.
    sliding_window : int or None
        If set, restrict each query to the most-recent ``sliding_window``
        keys.  ``None`` = full attention.
    softcap : float or None
        Gemma-2-style logit capping: ``softcap * tanh(score / softcap)``.
    num_kv_heads : int or None
        Number of KV heads for GQA/MQA.  ``None`` = same as query heads
        (standard MHA).
    compute_dtype : torch.dtype or None
        Lower-precision dtype for the Q·K^T matmul.  ``None`` = same as
        input.  Only used by mixed-precision path.
    use_triton : bool
        Prefer Triton GPU kernels when available.
    two_pass : bool
        Use two-pass softmax for better numerical precision.
    pipeline_stages : int
        Number of fine-grained pipeline stages (FFPA-style).
        1 = no pipelining; 2 = double-buffer; 3 = triple-buffer.
    return_lse : bool
        Return per-row logsumexp alongside the output.
    dropout_p : float
        Attention dropout probability (training only).
    deterministic : bool
        Use deterministic algorithms (may be slower).
    max_seqlen_q : int
        Maximum query sequence length (for static allocation).
    max_seqlen_kv : int
        Maximum key/value sequence length.
    page_size : int
        Page size for paged KV-cache.
    """

    block_size_q: int = 0
    block_size_kv: int = 0
    causal: bool = False
    sliding_window: Optional[int] = None
    softcap: Optional[float] = None
    num_kv_heads: Optional[int] = None
    compute_dtype: Optional[torch.dtype] = None
    use_triton: bool = True
    two_pass: bool = False
    pipeline_stages: int = 2
    return_lse: bool = False
    dropout_p: float = 0.0
    deterministic: bool = False
    max_seqlen_q: int = 0
    max_seqlen_kv: int = 0
    page_size: int = 64

    def resolve_block_sizes(self, S_q: int, S_kv: int, D: int) -> Tuple[int, int]:
        """Return (block_q, block_kv), auto-selecting if either is 0."""
        from .utils import auto_select_block_size

        bq = self.block_size_q or auto_select_block_size(S_q, S_kv, D)
        bkv = self.block_size_kv or auto_select_block_size(S_q, S_kv, D)
        return bq, bkv
