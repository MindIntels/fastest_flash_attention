"""
FastestFlashGQA — Grouped-Query Attention with Fastest Flash Attention.

Supports both GQA (num_kv_heads < num_heads) and MQA (num_kv_heads = 1).
KV heads are transparently repeated to match query heads before the
flash attention kernel.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..ops.attention_forward import fastest_flash_attn_forward
from ..ops.attention_func import fastest_flash_attn_func
from ..utils import repeat_kv


class FastestFlashGQA(nn.Module):
    """Grouped-Query Attention with Fastest Flash Attention.

    Parameters
    ----------
    hidden_size : int
        Model hidden dimension.
    num_heads : int
        Number of query heads.
    num_kv_heads : int
        Number of key/value heads (< num_heads for GQA, 1 for MQA).
    head_dim : int or None
        Dimension per head.
    bias : bool
        Bias in projection layers.
    causal : bool
        Default causal masking.
    block_size : int
        Tile size (0 = auto).
    softcap : float or None
        Logit capping.
    sliding_window : int or None
        Sliding window width.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: Optional[int] = None,
        bias: bool = False,
        causal: bool = False,
        block_size: int = 0,
        softcap: Optional[float] = None,
        sliding_window: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim or hidden_size // num_heads
        self.default_causal = causal
        self.block_size = block_size
        self.softcap = softcap
        self.sliding_window = sliding_window

        assert num_heads % num_kv_heads == 0, (
            "num_heads must be divisible by num_kv_heads"
        )

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=bias)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        causal: Optional[bool] = None,
        block_size: Optional[int] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_lse: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, S, hidden_size]
        Returns:
            [B, S, hidden_size]
        """
        B, S, _ = x.shape
        use_causal = causal if causal is not None else self.default_causal
        bs = block_size or self.block_size or None

        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Repeat KV heads to match query heads
        k = repeat_kv(k, self.num_heads, self.num_kv_heads)
        v = repeat_kv(v, self.num_heads, self.num_kv_heads)

        if self.training and not return_lse and kv_cache is None:
            attn_out = fastest_flash_attn_func(
                q, k, v,
                scale=self.scale,
                causal=use_causal,
                block_size=bs or 64,
                softcap=self.softcap,
                sliding_window=self.sliding_window,
            )
            lse = None
        else:
            attn_out, lse = fastest_flash_attn_forward(
                q, k, v,
                scale=self.scale,
                causal=use_causal,
                block_size=bs,
                softcap=self.softcap,
                sliding_window=self.sliding_window,
                kv_cache=kv_cache,
                return_lse=return_lse,
            )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, -1)
        out = self.o_proj(attn_out)

        if return_lse:
            return out, lse
        return out

    def extra_repr(self) -> str:
        return (
            f"hidden={self.hidden_size}, q_heads={self.num_heads}, "
            f"kv_heads={self.num_kv_heads}, head_dim={self.head_dim}, "
            f"causal={self.default_causal}, softcap={self.softcap}"
        )
