"""
Utility helpers for Fastest Flash Attention.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch


# ------------------------------------------------------------------
#  Triton availability check
# ------------------------------------------------------------------

_TRITON_AVAILABLE: bool | None = None


def check_triton_available() -> bool:
    """Return True if Triton is importable and a CUDA device exists."""
    global _TRITON_AVAILABLE
    if _TRITON_AVAILABLE is None:
        try:
            import triton  # noqa: F401
            _TRITON_AVAILABLE = torch.cuda.is_available()
        except ImportError:
            _TRITON_AVAILABLE = False
    return _TRITON_AVAILABLE


# ------------------------------------------------------------------
#  Adaptive block-size selection (FFPA-inspired heuristic)
# ------------------------------------------------------------------

def auto_select_block_size(
    S_q: int,
    S_kv: int,
    D: int,
    max_block: int = 256,
    min_block: int = 16,
) -> int:
    """Choose block size automatically based on problem geometry.

    Heuristic (inspired by FFPA tile-size selection):
      - Keep ``block * D`` within a register-friendly budget (~4096 elems).
      - Ensure ≥ 4 KV tiles for fine-grained pipeline benefit.
      - Prefer power-of-2 sizes for hardware alignment.
      - Smaller D → larger blocks (more reuse), larger D → smaller blocks.

    Returns
    -------
    int
        Block size in ``[min_block, max_block]``, always a power-of-2.
    """
    budget = 4096  # target register budget in elements
    bs = max(min_block, min(budget // max(D, 1), max_block))

    # ensure ≥ 4 KV tiles for pipelining
    while bs > min_block and math.ceil(S_kv / bs) < 4:
        bs //= 2

    # round down to power-of-2
    if bs > 0:
        bs = 1 << (bs.bit_length() - 1)

    return max(min_block, min(bs, max_block))


# ------------------------------------------------------------------
#  Numeric helpers
# ------------------------------------------------------------------

def safe_softmax(
    x: torch.Tensor,
    dim: int = -1,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Numerically-stable softmax: ``exp(x - max) / sum(exp(x - max))``."""
    if dtype is not None:
        x = x.to(dtype)
    x_max = x.max(dim=dim, keepdim=True).values
    x_stable = x - x_max
    exp_x = torch.exp(x_stable)
    out = exp_x / exp_x.sum(dim=dim, keepdim=True)
    return torch.nan_to_num(out, nan=0.0)


def logaddexp_safe(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Element-wise log-add-exp, handles -inf correctly."""
    return torch.logaddexp(a, b)


# ------------------------------------------------------------------
#  Shape helpers
# ------------------------------------------------------------------

def maybe_unsqueeze_head(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Ensure [B, H, S, D] layout; return tensors + squeezed flag."""
    squeezed = False
    if q.dim() == 3:
        q = q.unsqueeze(1)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)
        squeezed = True
    return q, k, v, squeezed


def repeat_kv(
    kv: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
) -> torch.Tensor:
    """Repeat KV heads to match query heads for GQA/MQA.

    Args:
        kv: [B, num_kv_heads, S, D]
        num_q_heads: total query heads.
        num_kv_heads: total kv heads.

    Returns:
        [B, num_q_heads, S, D]
    """
    if num_q_heads == num_kv_heads:
        return kv
    ratio = num_q_heads // num_kv_heads
    B, H_kv, S, D = kv.shape
    return kv[:, :, None, :, :].expand(B, H_kv, ratio, S, D).reshape(B, num_q_heads, S, D)
