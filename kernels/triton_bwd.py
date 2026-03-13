"""
Triton Flash Attention Backward Kernel.

Memory-efficient backward pass: recomputes S = Q·K^T from stored Q, K, V
and logsumexp, avoiding materialisation of the full attention matrix.

Gradients computed tile-by-tile using the same block decomposition as forward.
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
    def _flash_attn_bwd_kernel(
        Q, K, V, Out, dOut, LSE, dQ, dK, dV,
        stride_b, stride_s, stride_d,
        S_q: tl.constexpr, S_kv: tl.constexpr, D: tl.constexpr,
        scale: tl.constexpr,
        BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr,
        CAUSAL: tl.constexpr,
    ):
        """Backward kernel — one program per KV-block × batch·head."""
        pid_kv = tl.program_id(0)
        pid_bh = tl.program_id(1)

        k_start = pid_kv * BLOCK_KV
        offs_kv = k_start + tl.arange(0, BLOCK_KV)
        offs_d = tl.arange(0, D)

        # Load K, V block
        kv_mask = offs_kv[:, None] < S_kv
        k_ptrs = K + pid_bh * stride_b + offs_kv[:, None] * stride_s + offs_d[None, :] * stride_d
        v_ptrs = V + pid_bh * stride_b + offs_kv[:, None] * stride_s + offs_d[None, :] * stride_d
        k_block = tl.load(k_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
        v_block = tl.load(v_ptrs, mask=kv_mask, other=0.0).to(tl.float32)

        # Accumulators for gradients
        dk_acc = tl.zeros([BLOCK_KV, D], dtype=tl.float32)
        dv_acc = tl.zeros([BLOCK_KV, D], dtype=tl.float32)

        # Iterate over Q blocks
        q_end = S_q if not CAUSAL else min(S_q, k_start + BLOCK_KV + BLOCK_Q)
        for q_start in range(0, q_end, BLOCK_Q):
            offs_q = q_start + tl.arange(0, BLOCK_Q)

            # Skip if this Q block can't attend to this KV block
            if CAUSAL:
                if q_start < k_start:
                    if q_start + BLOCK_Q <= k_start:
                        continue

            q_mask = offs_q[:, None] < S_q

            # Load Q, dO, O, LSE
            q_ptrs = Q + pid_bh * stride_b + offs_q[:, None] * stride_s + offs_d[None, :] * stride_d
            do_ptrs = dOut + pid_bh * stride_b + offs_q[:, None] * stride_s + offs_d[None, :] * stride_d
            o_ptrs = Out + pid_bh * stride_b + offs_q[:, None] * stride_s + offs_d[None, :] * stride_d

            q_block = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)
            do_block = tl.load(do_ptrs, mask=q_mask, other=0.0).to(tl.float32)
            o_block = tl.load(o_ptrs, mask=q_mask, other=0.0).to(tl.float32)

            # LSE
            lse_ptrs = LSE + pid_bh * S_q + offs_q
            lse_mask = offs_q < S_q
            lse_block = tl.load(lse_ptrs, mask=lse_mask, other=0.0).to(tl.float32)

            # Recompute S
            s_ij = tl.dot(q_block, tl.trans(k_block)) * scale

            if CAUSAL:
                causal_mask = offs_q[:, None] >= offs_kv[None, :]
                s_ij = tl.where(causal_mask, s_ij, float("-inf"))

            valid_mask = (offs_q[:, None] < S_q) & (offs_kv[None, :] < S_kv)
            s_ij = tl.where(valid_mask, s_ij, float("-inf"))

            # P = exp(S - LSE)
            p_ij = tl.exp(s_ij - lse_block[:, None])

            # D_i = rowsum(dO * O)
            d_i = tl.sum(do_block * o_block, axis=1)

            # dV += P^T @ dO
            dv_acc += tl.dot(tl.trans(p_ij.to(do_block.dtype)), do_block)

            # dP = dO @ V^T
            dp_ij = tl.dot(do_block, tl.trans(v_block))

            # dS = P * (dP - D_i)
            ds_ij = p_ij * (dp_ij - d_i[:, None])

            # dK += dS^T @ Q * scale
            dk_acc += tl.dot(tl.trans(ds_ij.to(q_block.dtype)), q_block) * scale

            # dQ (atomic add)
            dq_block = tl.dot(ds_ij.to(k_block.dtype), k_block) * scale
            dq_ptrs = dQ + pid_bh * stride_b + offs_q[:, None] * stride_s + offs_d[None, :] * stride_d
            tl.atomic_add(dq_ptrs, dq_block.to(dQ.dtype.element_ty), mask=q_mask)

        # Store dK, dV
        dk_ptrs = dK + pid_bh * stride_b + offs_kv[:, None] * stride_s + offs_d[None, :] * stride_d
        dv_ptrs = dV + pid_bh * stride_b + offs_kv[:, None] * stride_s + offs_d[None, :] * stride_d
        tl.store(dk_ptrs, dk_acc.to(dK.dtype.element_ty), mask=kv_mask)
        tl.store(dv_ptrs, dv_acc.to(dV.dtype.element_ty), mask=kv_mask)


def flash_attn_triton_backward(
    grad_output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    scale: Optional[float] = None,
    causal: bool = False,
    block_q: int = 64,
    block_kv: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch Triton flash attention backward kernel.

    Args:
        grad_output, q, k, v, output: [B, H, S, D] on CUDA.
        lse: [B, H, S, 1] logsumexp from forward.

    Returns:
        (dq, dk, dv)
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton not available")

    B, H, S_q, D = q.shape
    S_kv = k.size(2)

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    D_padded = triton.next_power_of_2(D)
    pad = D_padded - D

    def maybe_pad(t):
        return torch.nn.functional.pad(t, (0, pad)) if pad > 0 else t

    q_p = maybe_pad(q).reshape(B * H, S_q, D_padded)
    k_p = maybe_pad(k).reshape(B * H, S_kv, D_padded)
    v_p = maybe_pad(v).reshape(B * H, S_kv, D_padded)
    o_p = maybe_pad(output).reshape(B * H, S_q, D_padded)
    do_p = maybe_pad(grad_output).reshape(B * H, S_q, D_padded)
    lse_flat = lse.squeeze(-1).reshape(B * H, S_q)

    dq = torch.zeros_like(q_p)
    dk = torch.zeros_like(k_p)
    dv = torch.zeros_like(v_p)

    n_kv_blocks = triton.cdiv(S_kv, block_kv)
    grid = (n_kv_blocks, B * H)

    _flash_attn_bwd_kernel[grid](
        q_p, k_p, v_p, o_p, do_p, lse_flat, dq, dk, dv,
        stride_b=q_p.stride(0), stride_s=q_p.stride(1), stride_d=q_p.stride(2),
        S_q=S_q, S_kv=S_kv, D=D_padded,
        scale=scale,
        BLOCK_Q=block_q, BLOCK_KV=block_kv,
        CAUSAL=causal,
    )

    dq = dq[:, :, :D].reshape(B, H, S_q, D)
    dk = dk[:, :, :D].reshape(B, H, S_kv, D)
    dv = dv[:, :, :D].reshape(B, H, S_kv, D)

    return dq, dk, dv
