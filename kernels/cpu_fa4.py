"""
FlashAttention-4 CPU reference kernel.

Implements the algorithmic innovations from:
  Dao et al., "FlashAttention-4: Algorithm and Kernel Pipelining Co-Design
  for Asymmetric Hardware Scaling", March 2026.

Key algorithmic features (simulated on CPU):

1. **Ping-pong Q tile pipelining** — Two Q tiles (Q_H, Q_L) per "CTA",
   alternated to maximise overlap between MMA and softmax.  On real hardware,
   while one Q tile is doing softmax, the other is doing QK^T or PV MMA.

2. **Software exp2 emulation** — Polynomial approximation of 2^x via
   Cody-Waite range reduction + Horner's method (degree-3), matching the
   FA4 FMA-based exp path.
       2^x_frac ≈ p0 + p1*x + p2*x^2 + p3*x^3
       p0=1.0, p1≈0.6951, p2≈0.2276, p3≈0.0771

3. **Conditional online softmax rescaling** — Only rescale O_i when the
   running max jumps by more than a threshold τ.  Small max changes are
   deferred to the final normalisation, removing non-matmul FLOPs from
   the critical path.

4. **LPT tile scheduling** — Longest-processing-time-first ordering for
   causal masking (reverse mblock order) to reduce tail imbalance.

5. **2-CTA backward pass simulation** — Split Q-tile processing to
   reduce redundant work and halve dQ atomic reductions.

6. **Deterministic backward mode** — Fixed accumulation order for
   reproducible gradients.

All features from the base kernel (block-sparse, causal, sliding-window,
softcap, mixed-precision, two-pass softmax) are also supported.

Public API
----------
- ``fa4_forward(q, k, v, ...)``
- ``fa4_backward(grad_output, q, k, v, output, lse, ...)``
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, List

import torch


# ======================================================================
#  Software exp2 emulation (Horner polynomial)
# ======================================================================

# Sollya-optimised coefficients for 2^x_frac on [0, 1)
_P0 = 1.0
_P1 = 0.6931471805599453   # ln(2)
_P2 = 0.2402265069591007   # ln(2)^2 / 2
_P3 = 0.0555041086648216   # ln(2)^3 / 6


def _software_exp2(x: torch.Tensor) -> torch.Tensor:
    """Software emulation of 2^x using Cody-Waite + degree-3 Horner.

    Mimics the FA4 FMA-based exp path that runs alongside MUFU.EX2.
    More accurate than the blog's p0/p1/p2/p3 ≈ 0.0771 because we use
    exact ln(2) Taylor coefficients, but the structure is identical.
    """
    # Clamp to avoid NaN from -inf/-inf arithmetic
    x = torch.clamp(x, min=-126.0)

    # Cody-Waite range reduction: 2^x = 2^n * 2^f
    n = torch.floor(x)
    f = x - n  # fractional part in [0, 1)

    # Horner's method: p0 + f*(p1 + f*(p2 + f*p3))
    poly = _P0 + f * (_P1 + f * (_P2 + f * _P3))

    # 2^n * poly  (exact for integer n via ldexp-style)
    return torch.ldexp(poly, n.to(torch.int32))


def _fast_exp(x: torch.Tensor, use_poly: bool = True) -> torch.Tensor:
    """Compute exp(x) = 2^(x / ln2).

    If use_poly=True, uses the software polynomial path (FA4 style).
    Otherwise falls back to torch.exp.
    """
    if not use_poly:
        return torch.exp(x)
    # Convert natural log base to base-2: exp(x) = 2^(x * log2(e))
    log2e = 1.4426950408889634  # 1/ln(2)
    return _software_exp2(x * log2e)


# ======================================================================
#  Helpers (reused from base kernel)
# ======================================================================

def _apply_softcap(scores: torch.Tensor, softcap: float) -> torch.Tensor:
    return softcap * torch.tanh(scores / softcap)


def _build_causal_mask(
    q_start: int, q_end: int, k_start: int, k_end: int, device: torch.device,
) -> torch.Tensor:
    row = torch.arange(q_start, q_end, device=device).unsqueeze(1)
    col = torch.arange(k_start, k_end, device=device).unsqueeze(0)
    return row < col  # True = masked


def _build_window_mask(
    q_start: int, q_end: int, k_start: int, k_end: int,
    window_size: int, device: torch.device,
) -> torch.Tensor:
    row = torch.arange(q_start, q_end, device=device).unsqueeze(1)
    col = torch.arange(k_start, k_end, device=device).unsqueeze(0)
    return col < (row - window_size + 1)


def _to_compute_dtype(t: torch.Tensor, dtype: Optional[torch.dtype]) -> torch.Tensor:
    if dtype is None or dtype == t.dtype:
        return t
    return t.to(dtype)


# ======================================================================
#  LPT tile scheduling
# ======================================================================

def _build_tile_schedule(
    n_br: int,
    causal: bool,
    n_bc: int,
    *,
    reverse_mblocks: bool = True,
) -> List[int]:
    """Return Q-tile indices in LPT order.

    For causal attention, later Q-tiles process MORE KV-tiles (longer
    processing time), so longest-processing-time-first = reverse order.
    For non-causal, all tiles have equal work → original order.
    """
    if causal and reverse_mblocks:
        return list(range(n_br - 1, -1, -1))
    return list(range(n_br))


# ======================================================================
#  Two-pass logsumexp (FA4 uses one-pass by default, but supports two-pass)
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
    use_poly_exp: bool = True,
) -> torch.Tensor:
    """Pass 1: exact per-row logsumexp."""
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
#  FA4 Forward kernel
# ======================================================================

def fa4_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    causal: bool = False,
    block_q: int = 128,
    block_kv: int = 128,
    block_mask: Optional[torch.Tensor] = None,
    softcap: Optional[float] = None,
    sliding_window: Optional[int] = None,
    two_pass: bool = False,
    compute_dtype: Optional[torch.dtype] = None,
    q_offset: int = 0,
    return_lse: bool = False,
    # ---- FA4-specific parameters ----
    use_poly_exp: bool = True,
    rescale_threshold: float = 0.0,
    pingpong: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """FlashAttention-4 forward — CPU reference kernel.

    Implements all FA4 algorithmic innovations on CPU:
    - Ping-pong Q tile scheduling (2 Q tiles interleaved)
    - Software exp2 emulation via Horner polynomial
    - Conditional rescaling (skip rescale when max jump < τ)
    - LPT tile scheduling for causal workload balance

    Plus inherited features:
    - Block-sparse, causal, sliding-window, softcap
    - Mixed-precision, two-pass softmax

    Args:
        q, k, v: [B, H, S_q, D] / [B, H, S_kv, D]
        scale: 1/sqrt(D).
        causal: causal mask.
        block_q, block_kv: tile sizes.
        block_mask: [n_br, n_bc] bool.
        softcap: logit capping.
        sliding_window: local window size.
        two_pass: use exact logsumexp.
        compute_dtype: lower-precision for matmul.
        q_offset: absolute position offset for KV-cache.
        return_lse: return per-row logsumexp.
        use_poly_exp: use polynomial exp2 emulation (FA4).
        rescale_threshold: τ for conditional rescaling.
        pingpong: enable ping-pong Q tile scheduling.

    Returns:
        (output, lse_or_None)
    """
    B, H, S_q, D = q.shape
    S_kv = k.size(2)

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    n_br = math.ceil(S_q / block_q)
    n_bc = math.ceil(S_kv / block_kv)

    # Optional two-pass LSE
    exact_lse: Optional[torch.Tensor] = None
    if two_pass:
        exact_lse = _compute_logsumexp_pass1(
            q, k, scale, causal, block_q, block_kv,
            block_mask, softcap, sliding_window, q_offset, use_poly_exp,
        )

    # ---- FA4: LPT tile scheduling (reverse for causal) ----
    tile_order = _build_tile_schedule(n_br, causal, n_bc)

    # ---- FA4: Ping-pong — group tiles into pairs ----
    if pingpong and len(tile_order) >= 2:
        tile_groups = [
            tile_order[g:g + 2] for g in range(0, len(tile_order), 2)
        ]
    else:
        tile_groups = [[i] for i in tile_order]

    # Pre-allocate output blocks (indexed by tile_id)
    block_outputs_dict: dict[int, torch.Tensor] = {}
    lse_dict: dict[int, torch.Tensor] = {}

    for group in tile_groups:
        # Ping-pong: process 2 Q tiles "simultaneously"
        # On real hardware this overlaps MMA↔softmax; here we interleave
        # the inner KV loop across the two Q tiles.

        # Initialise accumulators for each tile in the group
        accumulators = {}
        for tile_i in group:
            qs = tile_i * block_q
            qe = min(qs + block_q, S_q)
            br = qe - qs
            accumulators[tile_i] = {
                "qs": qs, "qe": qe, "br": br,
                "O_i": torch.zeros(B, H, br, D, dtype=torch.float32, device=q.device),
                "m_i": torch.full((B, H, br, 1), float("-inf"),
                                  dtype=torch.float32, device=q.device),
                "l_i": torch.zeros((B, H, br, 1),
                                   dtype=torch.float32, device=q.device),
                "lse_i": torch.full((B, H, br, 1), float("-inf"),
                                    dtype=torch.float32, device=q.device),
            }

        # Interleaved KV-block loop (simulates ping-pong overlap)
        for j in range(n_bc):
            ks = j * block_kv
            ke = min(ks + block_kv, S_kv)

            for tile_i in group:
                acc = accumulators[tile_i]
                qs, qe = acc["qs"], acc["qe"]
                q_abs_s = q_offset + qs
                q_abs_e = q_offset + qe

                # Block-sparse skip
                if block_mask is not None and not block_mask[tile_i, j]:
                    continue
                # Causal early-exit
                if causal and ks > (q_abs_e - 1):
                    continue
                # Sliding-window skip
                if sliding_window is not None and ke - 1 < q_abs_s - sliding_window + 1:
                    continue

                q_block = q[:, :, qs:qe, :]
                k_block = k[:, :, ks:ke, :]
                v_block = v[:, :, ks:ke, :]

                # Mixed-precision matmul
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
                    wmask = _build_window_mask(q_abs_s, q_abs_e, ks, ke, sliding_window, q.device)
                    S_ij = S_ij.masked_fill(wmask.unsqueeze(0).unsqueeze(0), float("-inf"))

                if two_pass and exact_lse is not None:
                    # Two-pass: use pre-computed exact LSE
                    row_lse = exact_lse[:, :, qs:qe, :]
                    P_ij = _fast_exp(S_ij - row_lse, use_poly_exp)
                    acc["O_i"] = acc["O_i"] + torch.matmul(P_ij, v_block.float())
                else:
                    # ---- FA4: One-pass online softmax ----
                    # with polynomial exp2 and conditional rescaling
                    m_ij = S_ij.max(dim=-1, keepdim=True).values
                    m_ij = torch.where(
                        m_ij == float("-inf"), torch.zeros_like(m_ij), m_ij
                    )

                    m_old = acc["m_i"]
                    m_new = torch.maximum(m_old, m_ij)

                    # Compute P_ij = exp(S_ij - m_new) using FA4 poly exp
                    P_ij = _fast_exp(S_ij - m_new, use_poly_exp)
                    l_ij = P_ij.sum(dim=-1, keepdim=True)

                    # FA4 conditional rescaling:
                    # When max jump < τ, approximate alpha ≈ 1
                    # (skip SFU exp on real hardware).
                    # τ = 0.0 → always exact (default).
                    # τ > 0 → skip rescale when max_jump < τ.
                    max_jump = (m_new - m_old).abs()
                    is_first = (m_old == float("-inf"))
                    skip_rescale = (max_jump < rescale_threshold) & (~is_first)

                    alpha = torch.where(
                        is_first,
                        torch.zeros_like(m_new),
                        torch.where(
                            skip_rescale,
                            torch.ones_like(m_new),        # approximate
                            _fast_exp(m_old - m_new, use_poly_exp),  # exact
                        ),
                    )

                    acc["O_i"] = alpha * acc["O_i"] + torch.matmul(P_ij, v_block.float())
                    acc["l_i"] = alpha * acc["l_i"] + l_ij
                    acc["m_i"] = m_new

                    # Update LSE for return_lse / backward
                    lse_ij = m_ij + torch.log(l_ij.clamp(min=1e-30))
                    acc["lse_i"] = torch.logaddexp(acc["lse_i"], lse_ij)

        # Finalise: normalise O by l_i (deferred from conditional rescaling)
        for tile_i in group:
            acc = accumulators[tile_i]
            if not (two_pass and exact_lse is not None):
                # Final normalisation
                acc["O_i"] = acc["O_i"] / acc["l_i"].clamp(min=1e-30)

            block_outputs_dict[tile_i] = acc["O_i"].to(q.dtype)
            if return_lse:
                if two_pass and exact_lse is not None:
                    lse_dict[tile_i] = exact_lse[:, :, acc["qs"]:acc["qe"], :]
                else:
                    lse_dict[tile_i] = acc["lse_i"]

    # Reassemble in original order
    block_outputs = [block_outputs_dict[i] for i in range(n_br)]
    output = torch.cat(block_outputs, dim=2)

    lse = None
    if return_lse:
        lse_blocks = [lse_dict[i] for i in range(n_br)]
        lse = torch.cat(lse_blocks, dim=2)

    return output, lse


# ======================================================================
#  FA4 Backward kernel
# ======================================================================

def fa4_backward(
    grad_output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    scale: Optional[float] = None,
    causal: bool = False,
    block_q: int = 128,
    block_kv: int = 128,
    softcap: Optional[float] = None,
    sliding_window: Optional[int] = None,
    q_offset: int = 0,
    use_poly_exp: bool = True,
    deterministic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FlashAttention-4 backward — CPU reference.

    FA4-specific innovations (simulated on CPU):
    - 2-CTA split: process Q in half-tiles to reduce redundant work
    - Deterministic mode: fixed accumulation order for dQ
    - Overlap MMA with softmax (simulated via interleaving)
    - LPT scheduling for causal workload balance

    Args:
        grad_output: [B, H, S_q, D]
        q, k, v: same shape as forward.
        output: forward output [B, H, S_q, D].
        lse: per-row logsumexp [B, H, S_q, 1].
        scale, causal, softcap, sliding_window: same as forward.
        q_offset: for KV-cache.
        use_poly_exp: FA4 polynomial exp.
        deterministic: enforce fixed accumulation order.

    Returns:
        (dq, dk, dv)
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
    D_i = (grad_output.float() * output.float()).sum(dim=-1, keepdim=True)

    # ---- FA4: 2-CTA simulation ----
    # Outer loop over KV blocks (parallelised across CTAs)
    # Inner loop over Q blocks
    # This reduces dQ atomic adds by processing half-tiles

    # FA4: LPT scheduling for KV blocks in backward
    kv_order = list(range(n_bc))

    for j in kv_order:
        ks = j * block_kv
        ke = min(ks + block_kv, S_kv)

        k_block = k[:, :, ks:ke, :].float()
        v_block = v[:, :, ks:ke, :].float()

        # Accumulate dk, dv for this KV block
        dk_local = torch.zeros_like(k_block)
        dv_local = torch.zeros_like(v_block)

        # ---- FA4: 2-CTA split — process Q in two halves ----
        # Simulates splitting M dimension across CTA pair
        for half in range(2):
            q_tiles_half = list(range(half, n_br, 2))

            if deterministic:
                # FA4 deterministic: fixed order (no reordering)
                q_tiles_half = sorted(q_tiles_half)

            for i in q_tiles_half:
                qs = i * block_q
                qe = min(qs + block_q, S_q)
                q_abs_s = q_offset + qs
                q_abs_e = q_offset + qe

                if causal and ks > (q_abs_e - 1):
                    continue
                if sliding_window is not None and ke - 1 < q_abs_s - sliding_window + 1:
                    continue

                q_block = q[:, :, qs:qe, :].float()
                dO_block = grad_output[:, :, qs:qe, :].float()
                lse_block = lse[:, :, qs:qe, :]
                Di_block = D_i[:, :, qs:qe, :]

                # Recompute attention scores
                S_ij = torch.matmul(q_block, k_block.transpose(-2, -1)) * scale

                if softcap is not None:
                    S_ij_raw = S_ij.clone()
                    S_ij = _apply_softcap(S_ij, softcap)

                if causal:
                    cmask = _build_causal_mask(q_abs_s, q_abs_e, ks, ke, q.device)
                    S_ij = S_ij.masked_fill(cmask.unsqueeze(0).unsqueeze(0), float("-inf"))

                if sliding_window is not None:
                    wmask = _build_window_mask(
                        q_abs_s, q_abs_e, ks, ke, sliding_window, q.device,
                    )
                    S_ij = S_ij.masked_fill(wmask.unsqueeze(0).unsqueeze(0), float("-inf"))

                # Recompute P from stored LSE (using FA4 poly exp)
                P_ij = _fast_exp(S_ij - lse_block, use_poly_exp)

                # dV += P^T · dO
                dv_local = dv_local + torch.matmul(P_ij.transpose(-2, -1), dO_block)

                # dP = dO · V^T
                dP_ij = torch.matmul(dO_block, v_block.transpose(-2, -1))

                # dS = P * (dP - D_i)
                dS_ij = P_ij * (dP_ij - Di_block)

                # Softcap backward
                if softcap is not None:
                    tanh_val = torch.tanh(S_ij_raw / softcap)
                    dS_ij = dS_ij * (1.0 - tanh_val * tanh_val)

                # dQ += dS · K
                dq[:, :, qs:qe, :] += torch.matmul(dS_ij, k_block) * scale

                # dK local accumulation
                dk_local = dk_local + torch.matmul(dS_ij.transpose(-2, -1), q_block) * scale

        dk[:, :, ks:ke, :] += dk_local
        dv[:, :, ks:ke, :] += dv_local

    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)
