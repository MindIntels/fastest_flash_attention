"""
Block-sparse attention pattern generators.

Create boolean block masks ``[n_br, n_bc]`` for selective tile skipping.
``True`` = process this tile, ``False`` = skip (treat as zero attention).

These masks can be combined (AND/OR) to create complex sparse patterns.
"""

from __future__ import annotations

import math

import torch


def create_causal_block_mask(
    S_q: int,
    S_kv: int,
    block_q: int,
    block_kv: int,
    q_offset: int = 0,
) -> torch.Tensor:
    """Block mask for causal (lower-triangular) attention.

    A block (i, j) is True if any (q, k) within the block satisfies q >= k.

    Returns:
        [n_br, n_bc] bool tensor.
    """
    n_br = math.ceil(S_q / block_q)
    n_bc = math.ceil(S_kv / block_kv)
    mask = torch.zeros(n_br, n_bc, dtype=torch.bool)

    for i in range(n_br):
        q_end = q_offset + min((i + 1) * block_q, S_q) - 1  # last q in block
        for j in range(n_bc):
            k_start = j * block_kv
            # Block is active if at least one q >= k_start
            if q_end >= k_start:
                mask[i, j] = True
            else:
                break  # All subsequent j are also masked

    return mask


def create_sliding_window_block_mask(
    S_q: int,
    S_kv: int,
    block_q: int,
    block_kv: int,
    window_size: int,
    q_offset: int = 0,
) -> torch.Tensor:
    """Block mask for sliding-window attention.

    A block (i, j) is True if any (q, k) within it satisfies
    ``k >= q - window_size + 1`` AND ``k <= q`` (combined with causal).

    Returns:
        [n_br, n_bc] bool tensor.
    """
    n_br = math.ceil(S_q / block_q)
    n_bc = math.ceil(S_kv / block_kv)
    mask = torch.zeros(n_br, n_bc, dtype=torch.bool)

    for i in range(n_br):
        q_start = q_offset + i * block_q
        q_end = q_offset + min((i + 1) * block_q, S_q) - 1

        for j in range(n_bc):
            k_start = j * block_kv
            k_end = min((j + 1) * block_kv, S_kv) - 1

            # Window condition: k_end >= q_start - window + 1
            # and k_start <= q_end
            if k_end >= q_start - window_size + 1 and k_start <= q_end:
                mask[i, j] = True

    return mask


def create_local_block_mask(
    S_q: int,
    S_kv: int,
    block_q: int,
    block_kv: int,
    local_radius: int,
) -> torch.Tensor:
    """Block mask for local (banded) attention.

    Each query attends to keys within ``local_radius`` positions in
    either direction.

    Returns:
        [n_br, n_bc] bool tensor.
    """
    n_br = math.ceil(S_q / block_q)
    n_bc = math.ceil(S_kv / block_kv)
    mask = torch.zeros(n_br, n_bc, dtype=torch.bool)

    for i in range(n_br):
        q_center = i * block_q + block_q // 2
        for j in range(n_bc):
            k_center = j * block_kv + block_kv // 2
            if abs(q_center - k_center) <= local_radius + max(block_q, block_kv):
                mask[i, j] = True

    return mask


def combine_block_masks(*masks: torch.Tensor, mode: str = "and") -> torch.Tensor:
    """Combine multiple block masks.

    Args:
        *masks: boolean [n_br, n_bc] tensors (must be same shape).
        mode: 'and' (intersection) or 'or' (union).

    Returns:
        Combined [n_br, n_bc] bool tensor.
    """
    assert len(masks) >= 1
    result = masks[0]
    for m in masks[1:]:
        if mode == "and":
            result = result & m
        elif mode == "or":
            result = result | m
        else:
            raise ValueError(f"Unknown mode: {mode!r}")
    return result
