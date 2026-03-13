"""
Triton Flash Attention Forward Kernel — FlashAttention-2/3 style.

Standard tiled attention with online softmax on GPU using Triton.
This is the baseline GPU kernel; see ``triton_ffpa.py`` for the
fine-grained pipelined variant.

Features:
  - Tiled attention with online softmax & deferred rescaling.
  - Causal mask with block-level early-exit.
  - Sliding-window support.
  - Softcap logit capping.
  - Mixed-precision (FP16/BF16 compute, FP32 accumulation).
  - Returns optional per-row logsumexp.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _flash_attn_fwd_kernel(
        Q, K, V, Out, LSE,
        stride_qb, stride_qh, stride_qs, stride_qd,
        stride_kb, stride_kh, stride_ks, stride_kd,
        stride_vb, stride_vh, stride_vs, stride_vd,
        stride_ob, stride_oh, stride_os, stride_od,
        stride_lb, stride_lh, stride_ls,
        S_q: tl.constexpr, S_kv: tl.constexpr, D: tl.constexpr,
        scale: tl.constexpr,
        BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr,
        CAUSAL: tl.constexpr,
        HAS_SOFTCAP: tl.constexpr, softcap: tl.constexpr,
        HAS_SLIDING_WINDOW: tl.constexpr, window_size: tl.constexpr,
        RETURN_LSE: tl.constexpr,
    ):
        # Program IDs
        pid_q = tl.program_id(0)   # which Q block
        pid_bh = tl.program_id(1)  # batch * head index

        # Compute batch and head indices
        num_heads = tl.num_programs(1) // (tl.num_programs(1) // tl.num_programs(1))
        # Actually: pid_bh encodes (batch_idx * H + head_idx)
        # We'll compute offsets directly.

        # Offsets for this Q block
        q_start = pid_q * BLOCK_Q
        offs_q = q_start + tl.arange(0, BLOCK_Q)
        offs_d = tl.arange(0, D)

        # Base pointers
        q_ptrs = Q + pid_bh * stride_qh + offs_q[:, None] * stride_qs + offs_d[None, :] * stride_qd
        o_ptrs = Out + pid_bh * stride_oh + offs_q[:, None] * stride_os + offs_d[None, :] * stride_od

        # Load Q block
        q_mask = offs_q[:, None] < S_q
        q_block = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)

        # Accumulators
        o_acc = tl.zeros([BLOCK_Q, D], dtype=tl.float32)
        m_i = tl.full([BLOCK_Q], value=float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_Q], dtype=tl.float32)

        # Determine KV range
        if CAUSAL:
            kv_end = min(S_kv, q_start + BLOCK_Q)
        else:
            kv_end = S_kv

        kv_start = 0
        if HAS_SLIDING_WINDOW:
            kv_start = max(0, q_start - window_size + 1)

        # Iterate over KV blocks
        for k_start in range(kv_start, kv_end, BLOCK_KV):
            offs_kv = k_start + tl.arange(0, BLOCK_KV)

            # Load K, V
            k_ptrs = K + pid_bh * stride_kh + offs_kv[:, None] * stride_ks + offs_d[None, :] * stride_kd
            v_ptrs = V + pid_bh * stride_vh + offs_kv[:, None] * stride_vs + offs_d[None, :] * stride_vd
            kv_mask = offs_kv[:, None] < S_kv
            k_block = tl.load(k_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
            v_block = tl.load(v_ptrs, mask=kv_mask, other=0.0).to(tl.float32)

            # S = Q @ K^T * scale
            s_ij = tl.dot(q_block, tl.trans(k_block)) * scale  # [BLOCK_Q, BLOCK_KV]

            # Softcap
            if HAS_SOFTCAP:
                s_ij = softcap * tl.math.tanh(s_ij / softcap)

            # Causal mask
            if CAUSAL:
                causal_mask = offs_q[:, None] >= offs_kv[None, :]
                s_ij = tl.where(causal_mask, s_ij, float("-inf"))

            # Sliding window mask
            if HAS_SLIDING_WINDOW:
                window_mask = offs_kv[None, :] >= (offs_q[:, None] - window_size + 1)
                s_ij = tl.where(window_mask, s_ij, float("-inf"))

            # Out-of-bounds mask
            valid_mask = (offs_q[:, None] < S_q) & (offs_kv[None, :] < S_kv)
            s_ij = tl.where(valid_mask, s_ij, float("-inf"))

            # Online softmax update
            m_ij = tl.max(s_ij, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            p_ij = tl.exp(s_ij - m_new[:, None])

            l_i = alpha * l_i + tl.sum(p_ij, axis=1)
            o_acc = alpha[:, None] * o_acc + tl.dot(p_ij.to(v_block.dtype), v_block)
            m_i = m_new

        # Final normalisation (deferred rescaling)
        l_safe = tl.where(l_i == 0.0, 1.0, l_i)
        o_acc = o_acc / l_safe[:, None]

        # Store output
        tl.store(o_ptrs, o_acc.to(Out.dtype.element_ty), mask=q_mask)

        # Store logsumexp
        if RETURN_LSE:
            lse_val = m_i + tl.log(tl.where(l_i > 0, l_i, 1e-30))
            lse_ptrs = LSE + pid_bh * stride_lh + offs_q * stride_ls
            lse_mask = offs_q < S_q
            tl.store(lse_ptrs, lse_val.to(LSE.dtype.element_ty), mask=lse_mask)


def flash_attn_triton_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    causal: bool = False,
    block_q: int = 128,
    block_kv: int = 64,
    softcap: Optional[float] = None,
    sliding_window: Optional[int] = None,
    return_lse: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Launch Triton flash attention forward kernel.

    Args:
        q: [B, H, S_q, D] on CUDA.
        k: [B, H, S_kv, D]
        v: [B, H, S_kv, D]
        Others same as CPU reference.

    Returns:
        (output, lse_or_None)
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton not available")

    B, H, S_q, D = q.shape
    S_kv = k.size(2)

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    # Ensure D is power of 2 for Triton (pad if needed)
    D_padded = triton.next_power_of_2(D)
    if D_padded != D:
        q = torch.nn.functional.pad(q, (0, D_padded - D))
        k = torch.nn.functional.pad(k, (0, D_padded - D))
        v = torch.nn.functional.pad(v, (0, D_padded - D))

    output = torch.empty(B, H, S_q, D_padded, dtype=q.dtype, device=q.device)
    lse = torch.empty(B, H, S_q, dtype=torch.float32, device=q.device) if return_lse else torch.empty(0, device=q.device)

    # Reshape to merge B and H for the kernel
    q_flat = q.reshape(B * H, S_q, D_padded)
    k_flat = k.reshape(B * H, S_kv, D_padded)
    v_flat = v.reshape(B * H, S_kv, D_padded)
    o_flat = output.reshape(B * H, S_q, D_padded)
    lse_flat = lse.reshape(B * H, S_q) if return_lse else lse

    n_q_blocks = triton.cdiv(S_q, block_q)
    grid = (n_q_blocks, B * H)

    _flash_attn_fwd_kernel[grid](
        q_flat, k_flat, v_flat, o_flat, lse_flat,
        # Q strides
        q_flat.stride(0), 0, q_flat.stride(1), q_flat.stride(2),
        # K strides
        k_flat.stride(0), 0, k_flat.stride(1), k_flat.stride(2),
        # V strides
        v_flat.stride(0), 0, v_flat.stride(1), v_flat.stride(2),
        # O strides
        o_flat.stride(0), 0, o_flat.stride(1), o_flat.stride(2),
        # LSE strides
        lse_flat.stride(0) if return_lse else 0,
        0,
        lse_flat.stride(1) if return_lse else 0,
        S_q=S_q, S_kv=S_kv, D=D_padded,
        scale=scale,
        BLOCK_Q=block_q, BLOCK_KV=block_kv,
        CAUSAL=causal,
        HAS_SOFTCAP=(softcap is not None),
        softcap=softcap if softcap is not None else 0.0,
        HAS_SLIDING_WINDOW=(sliding_window is not None),
        window_size=sliding_window if sliding_window is not None else 0,
        RETURN_LSE=return_lse,
    )

    output = output[:, :, :, :D]  # trim padding
    lse_out = lse.unsqueeze(-1) if return_lse else None

    return output, lse_out
