"""
Fastest Flash Attention — Forward dispatch.

Automatically selects between:
  1. FFPA Triton kernel (fastest, GPU + Triton)
  2. Standard Triton Flash Attention (GPU + Triton, no autotune overhead)
  3. CPU reference kernel (always available)

The dispatch is transparent: callers get the same API regardless of backend.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

from ..config import AttentionConfig
from ..utils import check_triton_available, maybe_unsqueeze_head, repeat_kv


def fastest_flash_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    config: Optional[AttentionConfig] = None,
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
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Fastest Flash Attention forward — auto-dispatch to best kernel.

    Args:
        q: [B, S_q, D], [B, H, S_q, D], or [B, H_q, S_q, D] for GQA.
        k: [B, S_kv, D], [B, H, S_kv, D], or [B, H_kv, S_kv, D].
        v: same layout as k.
        config: AttentionConfig (overrides individual params if given).
        scale: 1/sqrt(D).
        causal: lower-triangular mask.
        block_size: tile size (auto-select if None).
        block_mask: [n_br, n_bc] bool tensor.
        softcap: logit capping.
        sliding_window: sliding attention window width.
        two_pass: two-pass softmax.
        compute_dtype: lower-precision for matmuls.
        num_kv_heads: for GQA (None = standard MHA).
        kv_cache: (K_cache, V_cache) from prior steps.
        return_lse: return per-row logsumexp.

    Returns:
        (output, lse_or_None)
        output: same shape as q.
        lse: [B, H, S_q, 1] if return_lse else None.
    """
    # --- Resolve config -----------------------------------------------
    if config is not None:
        causal = config.causal
        softcap = config.softcap
        sliding_window = config.sliding_window
        two_pass = config.two_pass
        compute_dtype = config.compute_dtype
        num_kv_heads = config.num_kv_heads
        return_lse = config.return_lse

    # --- Handle 3D / 4D input -----------------------------------------
    q, k, v, squeezed = maybe_unsqueeze_head(q, k, v)
    B, H_q, S_q, D = q.shape
    H_kv = k.size(1)

    # --- KV-cache append ----------------------------------------------
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

    # --- GQA: repeat KV heads -----------------------------------------
    if num_kv_heads is not None and H_q != H_kv:
        k = repeat_kv(k, H_q, H_kv)
        v = repeat_kv(v, H_q, H_kv)

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    # --- Block size ---------------------------------------------------
    if block_size is None:
        if config is not None:
            bq, bkv = config.resolve_block_sizes(S_q, S_kv, D)
        else:
            from ..utils import auto_select_block_size
            bq = bkv = auto_select_block_size(S_q, S_kv, D)
    else:
        bq = bkv = block_size

    # --- Dispatch to best kernel --------------------------------------
    use_triton = (
        check_triton_available()
        and q.is_cuda
        and block_mask is None  # Triton kernels don't yet support block_mask
        and not two_pass        # Two-pass only on CPU for now
    )

    if use_triton:
        try:
            from ..kernels.triton_ffpa import ffpa_attn_triton_forward
            output, lse = ffpa_attn_triton_forward(
                q, k, v,
                scale=scale,
                causal=causal,
                block_q=bq,
                block_kv=bkv,
                softcap=softcap,
                return_lse=return_lse,
            )
        except Exception:
            # Fallback to standard Triton
            from ..kernels.triton_fwd import flash_attn_triton_forward
            output, lse = flash_attn_triton_forward(
                q, k, v,
                scale=scale,
                causal=causal,
                block_q=bq,
                block_kv=bkv,
                softcap=softcap,
                sliding_window=sliding_window,
                return_lse=return_lse,
            )
    else:
        from ..kernels.cpu_reference import flash_attn_cpu_forward
        output, lse = flash_attn_cpu_forward(
            q, k, v,
            scale=scale,
            causal=causal,
            block_q=bq,
            block_kv=bkv,
            block_mask=block_mask,
            softcap=softcap,
            sliding_window=sliding_window,
            two_pass=two_pass,
            compute_dtype=compute_dtype,
            q_offset=q_offset,
            return_lse=return_lse,
        )

    # --- Restore shape ------------------------------------------------
    if squeezed:
        output = output.squeeze(1)
        if lse is not None:
            lse = lse.squeeze(1)

    return output, lse
