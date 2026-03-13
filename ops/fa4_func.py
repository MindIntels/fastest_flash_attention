"""
FlashAttention-4 — Autograd function + functional API.

Wraps fa4_forward / fa4_backward into a torch.autograd.Function
so FA4 is fully differentiable.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch.autograd import Function

from ..kernels.cpu_fa4 import fa4_forward, fa4_backward
from ..utils import maybe_unsqueeze_head, repeat_kv


class FA4AttnFunc(Function):
    """Autograd-compatible FA4 forward + backward.

    Saves Q, K, V, output, lse for memory-efficient recomputation.
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
        use_poly_exp: bool,
        rescale_threshold: float,
        pingpong: bool,
        deterministic: bool,
    ) -> torch.Tensor:
        output, lse = fa4_forward(
            q, k, v,
            scale=scale,
            causal=causal,
            block_q=block_size,
            block_kv=block_size,
            softcap=softcap,
            sliding_window=sliding_window,
            two_pass=two_pass,
            return_lse=True,
            use_poly_exp=use_poly_exp,
            rescale_threshold=rescale_threshold,
            pingpong=pingpong,
        )

        ctx.save_for_backward(q, k, v, output, lse)
        ctx.scale = scale
        ctx.causal = causal
        ctx.block_size = block_size
        ctx.softcap = softcap
        ctx.sliding_window = sliding_window
        ctx.use_poly_exp = use_poly_exp
        ctx.deterministic = deterministic
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q, k, v, output, lse = ctx.saved_tensors

        was_3d = False
        if q.dim() == 3:
            was_3d = True
            q = q.unsqueeze(1)
            k = k.unsqueeze(1)
            v = v.unsqueeze(1)
            output = output.unsqueeze(1)
            lse = lse.unsqueeze(1)
            grad_output = grad_output.unsqueeze(1)

        dq, dk, dv = fa4_backward(
            grad_output, q, k, v, output, lse,
            scale=ctx.scale,
            causal=ctx.causal,
            block_q=ctx.block_size,
            block_kv=ctx.block_size,
            softcap=ctx.softcap,
            sliding_window=ctx.sliding_window,
            use_poly_exp=ctx.use_poly_exp,
            deterministic=ctx.deterministic,
        )

        if was_3d:
            dq = dq.squeeze(1)
            dk = dk.squeeze(1)
            dv = dv.squeeze(1)

        # Gradients for all forward args (None for non-Tensor)
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None


def fa4_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    causal: bool = False,
    block_size: int = 64,
    softcap: Optional[float] = None,
    sliding_window: Optional[int] = None,
    two_pass: bool = False,
    use_poly_exp: bool = True,
    rescale_threshold: float = 0.0,
    pingpong: bool = True,
    deterministic: bool = False,
) -> torch.Tensor:
    """Differentiable FlashAttention-4 — functional API.

    Recommended entry point for training with FA4 optimisations.

    Args:
        q: [B, S_q, D] or [B, H, S_q, D]
        k, v: matching layout.
        scale: 1/sqrt(D).
        causal: causal mask.
        block_size: tile size.
        softcap: logit capping.
        sliding_window: local window size.
        two_pass: two-pass softmax.
        use_poly_exp: FA4 polynomial exp2 emulation.
        rescale_threshold: τ for conditional rescaling.
        pingpong: FA4 ping-pong Q tile scheduling.
        deterministic: FA4 deterministic backward.

    Returns:
        [B, S_q, D] or [B, H, S_q, D] — attention output.
    """
    return FA4AttnFunc.apply(
        q, k, v, scale, causal, block_size, softcap, sliding_window,
        two_pass, use_poly_exp, rescale_threshold, pingpong, deterministic,
    )


def fa4_forward_dispatch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: Optional[float] = None,
    causal: bool = False,
    block_size: Optional[int] = None,
    block_mask: Optional[torch.Tensor] = None,
    softcap: Optional[float] = None,
    sliding_window: Optional[int] = None,
    two_pass: bool = False,
    compute_dtype: Optional[torch.dtype] = None,
    num_kv_heads: Optional[int] = None,
    kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    return_lse: bool = False,
    use_poly_exp: bool = True,
    rescale_threshold: float = 0.0,
    pingpong: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """FlashAttention-4 forward — handles shapes + KV-cache + GQA.

    This is the inference/evaluation entry point with FA4 features.

    Args:
        All base attention args plus FA4-specific:
        use_poly_exp, rescale_threshold, pingpong.

    Returns:
        (output, lse_or_None)
    """
    q, k, v, squeezed = maybe_unsqueeze_head(q, k, v)
    B, H_q, S_q, D = q.shape
    H_kv = k.size(1)

    # KV-cache append
    q_offset = 0
    if kv_cache is not None:
        k_cache, v_cache = kv_cache
        if k_cache.dim() == 3:
            k_cache = k_cache.unsqueeze(1)
            v_cache = v_cache.unsqueeze(1)
        k = torch.cat([k_cache, k], dim=2)
        v = torch.cat([v_cache, v], dim=2)
        q_offset = k.size(2) - S_q

    S_kv = k.size(2)

    # GQA: repeat KV heads
    if num_kv_heads is not None and H_q != H_kv:
        k = repeat_kv(k, H_q, H_kv)
        v = repeat_kv(v, H_q, H_kv)

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    # Block size
    if block_size is None:
        from ..utils import auto_select_block_size
        block_size = auto_select_block_size(S_q, S_kv, D)

    output, lse = fa4_forward(
        q, k, v,
        scale=scale,
        causal=causal,
        block_q=block_size,
        block_kv=block_size,
        block_mask=block_mask,
        softcap=softcap,
        sliding_window=sliding_window,
        two_pass=two_pass,
        compute_dtype=compute_dtype,
        q_offset=q_offset,
        return_lse=return_lse,
        use_poly_exp=use_poly_exp,
        rescale_threshold=rescale_threshold,
        pingpong=pingpong,
    )

    if squeezed:
        output = output.squeeze(1)
        if lse is not None:
            lse = lse.squeeze(1)

    return output, lse
