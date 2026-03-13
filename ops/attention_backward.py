"""
Fastest Flash Attention — Backward dispatch.

Recomputes attention from Q, K, V and stored logsumexp for
memory-efficient gradient computation.  Auto-dispatches to
Triton or CPU kernel.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

from ..utils import check_triton_available


def fastest_flash_attn_backward(
    grad_output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    *,
    scale: Optional[float] = None,
    causal: bool = False,
    block_size: int = 64,
    softcap: Optional[float] = None,
    sliding_window: Optional[int] = None,
    q_offset: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward pass — auto-dispatch to GPU or CPU kernel.

    Args:
        grad_output: [B, H, S_q, D]
        q, k, v, output: same as forward.
        lse: [B, H, S_q, 1] logsumexp from forward.
        Others: same as forward.

    Returns:
        (dq, dk, dv)
    """
    use_triton = (
        check_triton_available()
        and q.is_cuda
        and softcap is None  # Triton bwd doesn't yet handle softcap
        and sliding_window is None
    )

    if use_triton:
        try:
            from ..kernels.triton_bwd import flash_attn_triton_backward
            return flash_attn_triton_backward(
                grad_output, q, k, v, output, lse,
                scale=scale,
                causal=causal,
                block_q=block_size,
                block_kv=block_size,
            )
        except Exception:
            pass

    # CPU fallback
    from ..kernels.cpu_reference import flash_attn_cpu_backward
    return flash_attn_cpu_backward(
        grad_output, q, k, v, output, lse,
        scale=scale,
        causal=causal,
        block_q=block_size,
        block_kv=block_size,
        softcap=softcap,
        sliding_window=sliding_window,
        q_offset=q_offset,
    )
