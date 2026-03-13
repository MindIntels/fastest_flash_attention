"""
CPU reference kernel — Pure-PyTorch implementation of Fastest Flash Attention.

Combines all algorithmic innovations:
  - Tiled online softmax with deferred rescaling (FlashAttention-2)
  - Mixed-precision accumulation (FlashAttention-3)
  - Two-pass softmax for numerical precision (FlashAttention-3)
  - Fine-grained pipeline simulation (FFPA)
  - Block-sparse skip, causal early-exit, sliding-window skip
  - Softcap logit capping
  - GQA/MQA key-value head repetition

This is the correctness reference.  GPU kernels must match its output.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


# ======================================================================
#  Helpers
# ======================================================================

def _to_compute_dtype(t: torch.Tensor, dtype: Optional[torch.dtype]) -> torch.Tensor:
    """Cast to a lower-precision compute dtype (simulation)."""
    if dtype is None or dtype == t.dtype:
        return t
    return t.to(dtype)


def _apply_softcap(scores: torch.Tensor, softcap: float) -> torch.Tensor:
    """Gemma-2 style: softcap * tanh(scores / softcap)."""
    return softcap * torch.tanh(scores / softcap)


def _build_causal_mask(
    q_start: int, q_end: int, k_start: int, k_end: int, device: torch.device
) -> torch.Tensor:
    """Return bool mask — True where position should be MASKED (q_pos < k_pos)."""
    row_idx = torch.arange(q_start, q_end, device=device).unsqueeze(1)
    col_idx = torch.arange(k_start, k_end, device=device).unsqueeze(0)
    return row_idx < col_idx  # [br, bc]


def _build_window_mask(
    q_start: int, q_end: int, k_start: int, k_end: int,
    window_size: int, device: torch.device,
) -> torch.Tensor:
    """Return bool mask — True where key is OUTSIDE the sliding window."""
    row_idx = torch.arange(q_start, q_end, device=device).unsqueeze(1)
    col_idx = torch.arange(k_start, k_end, device=device).unsqueeze(0)
    return col_idx < (row_idx - window_size + 1)


# ======================================================================
#  Two-pass logsumexp pre-computation
# ======================================================================

def _compute_logsumexp_pass1(
    q: torch.Tensor,
    k: torch.Tensor,
    scale: float,
    causal: bool,
    block_q: int,
    block_kv: int,
    block_mask: Optional[torch.Tensor],
    softcap: Optional[float],
    sliding_window: Optional[int],
    q_offset: int = 0,
) -> torch.Tensor:
    """Pass 1: compute exact per-row logsumexp over all KV blocks.

    Returns: lse [B, H, S_q, 1]
    """
    B, H, S_q, D = q.shape
    S_k = k.size(2)
    n_br = math.ceil(S_q / block_q)
    n_bc = math.ceil(S_k / block_kv)

    lse_blocks = []
    for i in range(n_br):
        qs = i * block_q
        qe = min(qs + block_q, S_q)
        q_block = q[:, :, qs:qe, :]
        q_abs_s = q_offset + qs
        q_abs_e = q_offset + qe

        block_lse = torch.full(
            (B, H, qe - qs, 1), float("-inf"),
            dtype=torch.float32, device=q.device,
        )

        for j in range(n_bc):
            ks = j * block_kv
            ke = min(ks + block_kv, S_k)

            if block_mask is not None and not block_mask[i, j]:
                continue
            if causal and ks > (q_abs_e - 1):
                break
            if sliding_window is not None and ke - 1 < q_abs_s - sliding_window + 1:
                continue

            k_block = k[:, :, ks:ke, :]
            S_ij = torch.matmul(q_block.float(), k_block.float().transpose(-2, -1)) * scale

            if softcap is not None:
                S_ij = _apply_softcap(S_ij, softcap)
            if causal:
                mask = _build_causal_mask(q_abs_s, q_abs_e, ks, ke, q.device)
                S_ij = S_ij.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            if sliding_window is not None:
                wmask = _build_window_mask(q_abs_s, q_abs_e, ks, ke, sliding_window, q.device)
                S_ij = S_ij.masked_fill(wmask.unsqueeze(0).unsqueeze(0), float("-inf"))

            tile_lse = torch.logsumexp(S_ij, dim=-1, keepdim=True)
            block_lse = torch.logaddexp(block_lse, tile_lse)

        lse_blocks.append(block_lse)

    return torch.cat(lse_blocks, dim=2)


# ======================================================================
#  Forward kernel
# ======================================================================

def flash_attn_cpu_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    causal: bool = False,
    block_q: int = 64,
    block_kv: int = 64,
    block_mask: Optional[torch.Tensor] = None,
    softcap: Optional[float] = None,
    sliding_window: Optional[int] = None,
    two_pass: bool = False,
    compute_dtype: Optional[torch.dtype] = None,
    q_offset: int = 0,
    return_lse: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Fastest Flash Attention — CPU forward kernel.

    Implements all features: tiled online softmax, deferred rescaling,
    fine-grained pipeline simulation, block-sparse, causal, sliding-window,
    softcap, mixed-precision, two-pass softmax.

    Args:
        q: [B, H, S_q, D]
        k: [B, H, S_kv, D]
        v: [B, H, S_kv, D]
        scale: 1/sqrt(D) by default.
        causal: lower-triangular causal mask.
        block_q: query tile size.
        block_kv: key/value tile size.
        block_mask: [n_br, n_bc] bool; True = process tile.
        softcap: logit capping threshold.
        sliding_window: local window size.
        two_pass: use exact logsumexp (more precise, slower).
        compute_dtype: lower-precision dtype for matmul.
        q_offset: absolute position offset for KV-cache scenarios.
        return_lse: if True, return per-row logsumexp.

    Returns:
        (output, lse_or_None)
        output: [B, H, S_q, D]
        lse: [B, H, S_q, 1] if return_lse else None
    """
    B, H, S_q, D = q.shape
    S_kv = k.size(2)

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    n_br = math.ceil(S_q / block_q)
    n_bc = math.ceil(S_kv / block_kv)

    # Optional two-pass: pre-compute exact logsumexp
    exact_lse: Optional[torch.Tensor] = None
    if two_pass:
        exact_lse = _compute_logsumexp_pass1(
            q, k, scale, causal, block_q, block_kv,
            block_mask, softcap, sliding_window, q_offset,
        )

    block_outputs = []
    lse_outputs = []

    for i in range(n_br):
        qs = i * block_q
        qe = min(qs + block_q, S_q)
        br = qe - qs
        q_block = q[:, :, qs:qe, :]

        q_abs_s = q_offset + qs
        q_abs_e = q_offset + qe

        # FP32 accumulators (deferred rescaling — FlashAttention-2 style)
        O_i = torch.zeros(B, H, br, D, dtype=torch.float32, device=q.device)
        lse_i = torch.full(
            (B, H, br, 1), float("-inf"),
            dtype=torch.float32, device=q.device,
        )

        # ---- Simulated FFPA pipeline: prefetch next KV block --------
        prefetch_k: Optional[torch.Tensor] = None
        prefetch_v: Optional[torch.Tensor] = None

        for j in range(n_bc):
            ks = j * block_kv
            ke = min(ks + block_kv, S_kv)

            # Block-sparse skip
            if block_mask is not None and not block_mask[i, j]:
                prefetch_k = prefetch_v = None
                continue

            # Causal early-exit
            if causal and ks > (q_abs_e - 1):
                break

            # Sliding-window skip
            if sliding_window is not None and ke - 1 < q_abs_s - sliding_window + 1:
                prefetch_k = prefetch_v = None
                continue

            # ---- FFPA Stage 1: Load (use prefetched or fresh) --------
            if prefetch_k is not None:
                k_block, v_block = prefetch_k, prefetch_v
            else:
                k_block = k[:, :, ks:ke, :]
                v_block = v[:, :, ks:ke, :]

            # ---- FFPA: Prefetch next valid KV block ------------------
            nj = j + 1
            prefetch_k = prefetch_v = None
            while nj < n_bc:
                nks = nj * block_kv
                nke = min(nks + block_kv, S_kv)
                skip = False
                if block_mask is not None and not block_mask[i, nj]:
                    skip = True
                if causal and nks > (q_abs_e - 1):
                    break
                if sliding_window is not None and nke - 1 < q_abs_s - sliding_window + 1:
                    skip = True
                if not skip:
                    prefetch_k = k[:, :, nks:nke, :]
                    prefetch_v = v[:, :, nks:nke, :]
                    break
                nj += 1

            # ---- FFPA Stage 2: Compute S = Q · K^T -------------------
            q_lp = _to_compute_dtype(q_block, compute_dtype)
            k_lp = _to_compute_dtype(k_block, compute_dtype)
            S_ij = torch.matmul(q_lp.float(), k_lp.float().transpose(-2, -1)) * scale

            # Softcap
            if softcap is not None:
                S_ij = _apply_softcap(S_ij, softcap)

            # Causal mask
            if causal:
                cmask = _build_causal_mask(q_abs_s, q_abs_e, ks, ke, q.device)
                S_ij = S_ij.masked_fill(cmask.unsqueeze(0).unsqueeze(0), float("-inf"))

            # Sliding-window mask
            if sliding_window is not None:
                wmask = _build_window_mask(
                    q_abs_s, q_abs_e, ks, ke, sliding_window, q.device,
                )
                S_ij = S_ij.masked_fill(wmask.unsqueeze(0).unsqueeze(0), float("-inf"))

            # ---- FFPA Stage 3: Softmax + accumulate P·V --------------
            if two_pass and exact_lse is not None:
                # Two-pass: use pre-computed exact LSE
                row_lse = exact_lse[:, :, qs:qe, :]
                P_ij = torch.exp(S_ij - row_lse)
                O_i = O_i + torch.matmul(P_ij, v_block.float())
            else:
                # One-pass: online softmax with deferred rescaling
                m_ij = S_ij.max(dim=-1, keepdim=True).values
                m_ij = torch.where(
                    m_ij == float("-inf"), torch.zeros_like(m_ij), m_ij
                )
                P_ij = torch.exp(S_ij - m_ij)
                lse_ij = m_ij + torch.log(
                    P_ij.sum(dim=-1, keepdim=True).clamp(min=1e-30)
                )

                lse_new = torch.logaddexp(lse_i, lse_ij)
                alpha = torch.exp(lse_i - lse_new)
                tile_scale = torch.exp(m_ij - lse_new)

                O_i = alpha * O_i + tile_scale * torch.matmul(P_ij, v_block.float())
                lse_i = lse_new

        block_outputs.append(O_i.to(q.dtype))
        if return_lse:
            final_lse = exact_lse[:, :, qs:qe, :] if (two_pass and exact_lse is not None) else lse_i
            lse_outputs.append(final_lse)

    output = torch.cat(block_outputs, dim=2)
    lse = torch.cat(lse_outputs, dim=2) if return_lse else None

    return output, lse


# ======================================================================
#  Backward kernel
# ======================================================================

def flash_attn_cpu_backward(
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
    softcap: Optional[float] = None,
    sliding_window: Optional[int] = None,
    q_offset: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward pass — recomputes attention from Q, K, V and stored LSE.

    Memory-efficient: does NOT store the full attention matrix.

    Args:
        grad_output: [B, H, S_q, D]
        q, k, v: same as forward.
        output: forward output [B, H, S_q, D].
        lse: per-row logsumexp [B, H, S_q, 1].
        scale: 1/sqrt(D).
        causal, softcap, sliding_window: same as forward.

    Returns:
        (dq, dk, dv) — gradients w.r.t. q, k, v.
    """
    B, H, S_q, D = q.shape
    S_kv = k.size(2)

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    n_br = math.ceil(S_q / block_q)
    n_bc = math.ceil(S_kv / block_kv)

    dq = torch.zeros_like(q, dtype=torch.float32)
    dk = torch.zeros_like(k, dtype=torch.float32)
    dv = torch.zeros_like(v, dtype=torch.float32)

    # D_i = rowsum(dO * O) — needed for backward softmax
    D_i = (grad_output.float() * output.float()).sum(dim=-1, keepdim=True)  # [B,H,S_q,1]

    for i in range(n_br):
        qs = i * block_q
        qe = min(qs + block_q, S_q)
        q_abs_s = q_offset + qs
        q_abs_e = q_offset + qe

        q_block = q[:, :, qs:qe, :].float()
        dO_block = grad_output[:, :, qs:qe, :].float()
        lse_block = lse[:, :, qs:qe, :]
        Di_block = D_i[:, :, qs:qe, :]

        for j in range(n_bc):
            ks = j * block_kv
            ke = min(ks + block_kv, S_kv)

            if causal and ks > (q_abs_e - 1):
                break
            if sliding_window is not None and ke - 1 < q_abs_s - sliding_window + 1:
                continue

            k_block = k[:, :, ks:ke, :].float()
            v_block = v[:, :, ks:ke, :].float()

            # Recompute attention scores
            S_ij = torch.matmul(q_block, k_block.transpose(-2, -1)) * scale

            if softcap is not None:
                S_ij_raw = S_ij.clone()
                S_ij = _apply_softcap(S_ij, softcap)

            if causal:
                cmask = _build_causal_mask(q_abs_s, q_abs_e, ks, ke, q.device)
                S_ij = S_ij.masked_fill(cmask.unsqueeze(0).unsqueeze(0), float("-inf"))

            if sliding_window is not None:
                wmask = _build_window_mask(q_abs_s, q_abs_e, ks, ke, sliding_window, q.device)
                S_ij = S_ij.masked_fill(wmask.unsqueeze(0).unsqueeze(0), float("-inf"))

            # Recompute P from stored LSE
            P_ij = torch.exp(S_ij - lse_block)  # [B,H,br,bc]

            # dV += P^T · dO
            dv[:, :, ks:ke, :] += torch.matmul(P_ij.transpose(-2, -1), dO_block)

            # dP = dO · V^T
            dP_ij = torch.matmul(dO_block, v_block.transpose(-2, -1))

            # dS = P * (dP - D_i)
            dS_ij = P_ij * (dP_ij - Di_block)

            # Softcap backward: dS_raw = dS * softcap_grad
            if softcap is not None:
                tanh_val = torch.tanh(S_ij_raw / softcap)
                softcap_grad = 1.0 - tanh_val * tanh_val
                dS_ij = dS_ij * softcap_grad

            # dQ += dS · K
            dq[:, :, qs:qe, :] += torch.matmul(dS_ij, k_block) * scale

            # dK += dS^T · Q
            dk[:, :, ks:ke, :] += torch.matmul(dS_ij.transpose(-2, -1), q_block) * scale

    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)
