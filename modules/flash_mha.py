"""
FastestFlashMHA — Multi-Head Attention with Fastest Flash Attention kernel.

Drop-in replacement for standard MHA with all flash attention optimisations.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..config import AttentionConfig
from ..ops.attention_forward import fastest_flash_attn_forward
from ..ops.attention_func import fastest_flash_attn_func


class FastestFlashMHA(nn.Module):
    """Multi-Head Attention powered by Fastest Flash Attention.

    Parameters
    ----------
    hidden_size : int
        Model hidden dimension (= num_heads * head_dim).
    num_heads : int
        Number of attention heads.
    head_dim : int or None
        Dimension per head.  Defaults to ``hidden_size // num_heads``.
    dropout : float
        Attention dropout (training only, not used in flash path).
    bias : bool
        Add bias to projection layers.
    causal : bool
        Default causal masking.
    block_size : int
        Default tile size (0 = auto).
    softcap : float or None
        Logit soft-capping.
    sliding_window : int or None
        Sliding window attention width.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
        causal: bool = False,
        block_size: int = 0,
        softcap: Optional[float] = None,
        sliding_window: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim or hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, (
            "hidden_size must equal num_heads * head_dim"
        )
        self.default_causal = causal
        self.block_size = block_size
        self.softcap = softcap
        self.sliding_window = sliding_window

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def load_weights(
        self,
        w_q: torch.Tensor,
        w_k: torch.Tensor,
        w_v: torch.Tensor,
        w_o: torch.Tensor,
    ):
        """Load pre-defined weight matrices."""
        self.q_proj.weight.data.copy_(w_q)
        self.k_proj.weight.data.copy_(w_k)
        self.v_proj.weight.data.copy_(w_v)
        self.o_proj.weight.data.copy_(w_o)

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
            causal: override default causal flag.
            block_size: override default block size.
            kv_cache: optional (K_cache, V_cache) for autoregressive.
            return_lse: return per-row logsumexp.

        Returns:
            [B, S, hidden_size]  or  (output, lse)
        """
        B, S, _ = x.shape
        use_causal = causal if causal is not None else self.default_causal
        bs = block_size or self.block_size or None

        # Project
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape: [B, S, H] → [B, num_heads, S, head_dim]
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Use autograd-compatible function for training
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

        # Reshape back
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, -1)
        out = self.o_proj(attn_out)

        if return_lse:
            return out, lse
        return out

    def extra_repr(self) -> str:
        return (
            f"hidden={self.hidden_size}, heads={self.num_heads}, "
            f"head_dim={self.head_dim}, block_size={self.block_size}, "
            f"causal={self.default_causal}, softcap={self.softcap}, "
            f"window={self.sliding_window}"
        )
