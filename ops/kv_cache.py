"""
KV-Cache implementations for autoregressive decoding.

Provides two strategies:
  - ContinuousKVCache: simple contiguous tensor, append-based.
  - PagedKVCache: paged memory management (vLLM-style) for
    memory-efficient multi-sequence serving.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


class ContinuousKVCache:
    """Simple contiguous KV-cache — append new K/V each step.

    Suitable for single-sequence inference or when memory is not
    the bottleneck.

    Args:
        max_batch: max batch size.
        max_seq_len: max total sequence length (past + new).
        num_heads: number of KV heads.
        head_dim: dimension per head.
        dtype: tensor dtype.
        device: device.
    """

    def __init__(
        self,
        max_batch: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: torch.device | str = "cpu",
    ):
        self.max_batch = max_batch
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Pre-allocate buffers
        self.k_cache = torch.zeros(
            max_batch, num_heads, max_seq_len, head_dim,
            dtype=dtype, device=device,
        )
        self.v_cache = torch.zeros(
            max_batch, num_heads, max_seq_len, head_dim,
            dtype=dtype, device=device,
        )
        # Track current length per batch element
        self.seq_lens = torch.zeros(max_batch, dtype=torch.long, device=device)

    def reset(self, batch_indices: Optional[torch.Tensor] = None):
        """Reset cache (all or specific batch entries)."""
        if batch_indices is None:
            self.seq_lens.zero_()
        else:
            self.seq_lens[batch_indices] = 0

    def append(
        self,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append new K/V and return full cached K, V.

        Args:
            k_new: [B, H, S_new, D]
            v_new: [B, H, S_new, D]

        Returns:
            (k_full, v_full): [B, H, S_total, D]
        """
        B, H, S_new, D = k_new.shape

        for b in range(B):
            start = self.seq_lens[b].item()
            end = start + S_new
            assert end <= self.max_seq_len, (
                f"KV-cache overflow: {end} > {self.max_seq_len}"
            )
            self.k_cache[b, :, start:end, :] = k_new[b]
            self.v_cache[b, :, start:end, :] = v_new[b]
            self.seq_lens[b] = end

        # Return views of valid cached data
        max_len = self.seq_lens[:B].max().item()
        return (
            self.k_cache[:B, :, :max_len, :],
            self.v_cache[:B, :, :max_len, :],
        )

    def get_cache(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return current cached K, V."""
        max_len = self.seq_lens[:batch_size].max().item()
        if max_len == 0:
            return None, None
        return (
            self.k_cache[:batch_size, :, :max_len, :],
            self.v_cache[:batch_size, :, :max_len, :],
        )


class PagedKVCache:
    """Paged KV-cache — vLLM-style page-table management.

    Memory is divided into fixed-size pages.  Each sequence has a page
    table mapping logical blocks to physical pages.  This enables
    non-contiguous storage and efficient memory sharing between
    sequences (e.g. prefix caching).

    Args:
        max_pages: total number of physical pages.
        page_size: tokens per page.
        num_heads: number of KV heads.
        head_dim: dimension per head.
        dtype: tensor dtype.
        device: device.
    """

    def __init__(
        self,
        max_pages: int = 1024,
        page_size: int = 64,
        num_heads: int = 8,
        head_dim: int = 64,
        dtype: torch.dtype = torch.float16,
        device: torch.device | str = "cpu",
    ):
        self.max_pages = max_pages
        self.page_size = page_size
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Physical page pool: [max_pages, num_heads, page_size, head_dim]
        self.k_pages = torch.zeros(
            max_pages, num_heads, page_size, head_dim,
            dtype=dtype, device=device,
        )
        self.v_pages = torch.zeros(
            max_pages, num_heads, page_size, head_dim,
            dtype=dtype, device=device,
        )

        # Free page tracking
        self.free_pages = list(range(max_pages))

        # Per-sequence state: {seq_id: {"page_table": [...], "len": int}}
        self.sequences: dict = {}

    def allocate_page(self) -> int:
        """Allocate a single physical page. Returns page_id."""
        if not self.free_pages:
            raise RuntimeError("PagedKVCache: out of physical pages")
        return self.free_pages.pop(0)

    def free_sequence(self, seq_id: int):
        """Free all pages belonging to a sequence."""
        if seq_id in self.sequences:
            for page_id in self.sequences[seq_id]["page_table"]:
                self.free_pages.append(page_id)
            del self.sequences[seq_id]

    def append_tokens(
        self,
        seq_id: int,
        k_tokens: torch.Tensor,
        v_tokens: torch.Tensor,
    ) -> int:
        """Append new K/V tokens to a sequence.

        Args:
            seq_id: unique sequence identifier.
            k_tokens: [num_heads, S_new, head_dim]
            v_tokens: [num_heads, S_new, head_dim]

        Returns:
            New total sequence length.
        """
        if seq_id not in self.sequences:
            self.sequences[seq_id] = {"page_table": [], "len": 0}

        seq = self.sequences[seq_id]
        S_new = k_tokens.size(1)

        for t in range(S_new):
            current_len = seq["len"]
            page_idx = current_len // self.page_size
            offset = current_len % self.page_size

            # Allocate new page if needed
            while len(seq["page_table"]) <= page_idx:
                seq["page_table"].append(self.allocate_page())

            phys_page = seq["page_table"][page_idx]
            self.k_pages[phys_page, :, offset, :] = k_tokens[:, t, :]
            self.v_pages[phys_page, :, offset, :] = v_tokens[:, t, :]
            seq["len"] += 1

        return seq["len"]

    def get_kv(self, seq_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather contiguous K, V for a sequence from its pages.

        Returns:
            (k, v): each [num_heads, seq_len, head_dim]
        """
        if seq_id not in self.sequences or self.sequences[seq_id]["len"] == 0:
            return None, None

        seq = self.sequences[seq_id]
        total_len = seq["len"]
        k_out = torch.empty(
            self.num_heads, total_len, self.head_dim,
            dtype=self.k_pages.dtype, device=self.k_pages.device,
        )
        v_out = torch.empty_like(k_out)

        pos = 0
        for page_id in seq["page_table"]:
            page_len = min(self.page_size, total_len - pos)
            k_out[:, pos:pos + page_len, :] = self.k_pages[page_id, :, :page_len, :]
            v_out[:, pos:pos + page_len, :] = self.v_pages[page_id, :, :page_len, :]
            pos += page_len

        return k_out, v_out

    @property
    def num_free_pages(self) -> int:
        return len(self.free_pages)

    @property
    def total_capacity(self) -> int:
        return self.max_pages * self.page_size
