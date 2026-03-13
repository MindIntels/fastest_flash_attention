"""
Fastest Flash Attention — Autograd function.

Wraps forward + backward into a ``torch.autograd.Function`` so that
``fastest_flash_attn_func`` is fully differentiable.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch.autograd import Function

from .attention_forward import fastest_flash_attn_forward
from .attention_backward import fastest_flash_attn_backward


class FastestFlashAttnFunc(Function):
    """Autograd-compatible forward + backward for Fastest Flash Attention.

    Saves only Q, K, V and logsumexp (not the full attention matrix),
    making it memory-efficient for long sequences.
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: Optional[float],
        causal: bool,
        block_size: int,
        softcap: Optional[float],
        sliding_window: Optional[int],
        two_pass: bool,
    ) -> torch.Tensor:
        # Always request LSE for backward
        output, lse = fastest_flash_attn_forward(
            q, k, v,
            scale=scale,
            causal=causal,
            block_size=block_size,
            softcap=softcap,
            sliding_window=sliding_window,
            two_pass=two_pass,
            return_lse=True,
        )

        ctx.save_for_backward(q, k, v, output, lse)
        ctx.scale = scale
        ctx.causal = causal
        ctx.block_size = block_size
        ctx.softcap = softcap
        ctx.sliding_window = sliding_window
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q, k, v, output, lse = ctx.saved_tensors

        # Ensure 4D for backward kernel
        was_3d = False
        if q.dim() == 3:
            was_3d = True
            q = q.unsqueeze(1)
            k = k.unsqueeze(1)
            v = v.unsqueeze(1)
            output = output.unsqueeze(1)
            lse = lse.unsqueeze(1)
            grad_output = grad_output.unsqueeze(1)

        dq, dk, dv = fastest_flash_attn_backward(
            grad_output, q, k, v, output, lse,
            scale=ctx.scale,
            causal=ctx.causal,
            block_size=ctx.block_size,
            softcap=ctx.softcap,
            sliding_window=ctx.sliding_window,
        )

        if was_3d:
            dq = dq.squeeze(1)
            dk = dk.squeeze(1)
            dv = dv.squeeze(1)

        # Return gradients for all forward args (None for non-Tensor)
        return dq, dk, dv, None, None, None, None, None, None


def fastest_flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    causal: bool = False,
    block_size: int = 64,
    softcap: Optional[float] = None,
    sliding_window: Optional[int] = None,
    two_pass: bool = False,
) -> torch.Tensor:
    """Differentiable fastest flash attention — functional API.

    This is the **recommended** entry point for training use.
    Supports backward pass with memory-efficient recomputation.

    Args:
        q: [B, S_q, D] or [B, H, S_q, D]
        k: [B, S_kv, D] or [B, H, S_kv, D]
        v: [B, S_kv, D] or [B, H, S_kv, D]
        scale: 1/sqrt(D).
        causal: causal mask.
        block_size: tile size.
        softcap: logit capping.
        sliding_window: local window size.
        two_pass: two-pass softmax for precision.

    Returns:
        Tensor with same shape as q.
    """
    return FastestFlashAttnFunc.apply(
        q, k, v, scale, causal, block_size, softcap, sliding_window, two_pass,
    )
