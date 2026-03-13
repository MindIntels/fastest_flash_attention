"""
FastestFlashMLA — Multi-Latent Attention with Fastest Flash Attention.

Implements the MLA (Multi-Latent Attention) mechanism used in DeepSeek-V2/V3:
  - Low-rank KV compression via learnable down-projection + up-projection.
  - Decoupled RoPE: position information is injected through a separate
    projection path rather than modifying the K/V directly.
  - Compatible with the Fastest Flash Attention kernel for O(1) memory per tile.

Architecture:
    Q → q_proj → [q_nope, q_rope] → combine with RoPE → Q_final
    X → kv_down_proj → latent → kv_up_proj → [K_nope, V] + k_rope → combine

This reduces KV-cache memory from O(2·H·D) to O(latent_dim) per token.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..ops.attention_forward import fastest_flash_attn_forward
from ..ops.attention_func import fastest_flash_attn_func


class RotaryEmbedding(nn.Module):
    """Simplified RoPE for the decoupled position path."""

    def __init__(self, dim: int, max_len: int = 8192, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_len = max_len

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Apply RoPE to x: [B, H, S, D_rope]."""
        S = x.size(2)
        t = torch.arange(offset, offset + S, device=x.device, dtype=x.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(x.device))
        emb = torch.cat([freqs, freqs], dim=-1)  # [S, D_rope]

        cos_emb = emb.cos().unsqueeze(0).unsqueeze(0)  # [1, 1, S, D_rope]
        sin_emb = emb.sin().unsqueeze(0).unsqueeze(0)

        # Rotate
        x1 = x[..., : x.size(-1) // 2]
        x2 = x[..., x.size(-1) // 2 :]
        return torch.cat([x1 * cos_emb[..., :x1.size(-1)] - x2 * sin_emb[..., :x1.size(-1)],
                          x2 * cos_emb[..., :x1.size(-1)] + x1 * sin_emb[..., :x1.size(-1)]], dim=-1)


class FastestFlashMLA(nn.Module):
    """Multi-Latent Attention (MLA) with Fastest Flash Attention.

    Parameters
    ----------
    hidden_size : int
        Model hidden dimension.
    num_heads : int
        Number of query heads.
    head_dim : int
        Dimension per head for attention.
    latent_dim : int
        Dimension of the compressed KV latent space.
    rope_dim : int
        Dimension for the decoupled RoPE path.
    causal : bool
        Default causal masking.
    block_size : int
        Tile size (0 = auto).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int = 128,
        latent_dim: int = 512,
        rope_dim: int = 64,
        causal: bool = True,
        block_size: int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.latent_dim = latent_dim
        self.rope_dim = rope_dim
        self.default_causal = causal
        self.block_size = block_size

        # Effective attention dimension: nope + rope
        self.attn_dim = head_dim + rope_dim
        self.scale = 1.0 / math.sqrt(self.attn_dim)

        # Query projections
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.q_rope_proj = nn.Linear(hidden_size, num_heads * rope_dim, bias=False)

        # KV compression: X → latent → K_nope, V
        self.kv_down_proj = nn.Linear(hidden_size, latent_dim, bias=False)
        self.kv_up_proj = nn.Linear(
            latent_dim, num_heads * (head_dim + head_dim), bias=False,
        )  # outputs K_nope and V concatenated

        # Decoupled RoPE for K
        self.k_rope_proj = nn.Linear(hidden_size, rope_dim, bias=False)
        self.rope = RotaryEmbedding(rope_dim)

        # Output projection
        self.o_proj = nn.Linear(num_heads * (head_dim + rope_dim), hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        causal: Optional[bool] = None,
        block_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, S, hidden_size]
        Returns:
            [B, S, hidden_size]
        """
        B, S, _ = x.shape
        use_causal = causal if causal is not None else self.default_causal
        bs = block_size or self.block_size or None

        # Q: nope + rope
        q_nope = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        q_rope = self.q_rope_proj(x).view(B, S, self.num_heads, self.rope_dim).transpose(1, 2)
        q_rope = self.rope(q_rope)
        q = torch.cat([q_nope, q_rope], dim=-1)  # [B, H, S, head_dim + rope_dim]

        # KV: compressed latent → up-project
        kv_latent = self.kv_down_proj(x)  # [B, S, latent_dim]
        kv = self.kv_up_proj(kv_latent)   # [B, S, H * (head_dim + head_dim)]
        kv = kv.view(B, S, self.num_heads, 2 * self.head_dim).transpose(1, 2)
        k_nope = kv[:, :, :, :self.head_dim]
        v = kv[:, :, :, self.head_dim:]

        # K: nope + rope (decoupled)
        k_rope_shared = self.k_rope_proj(x)  # [B, S, rope_dim]
        k_rope_shared = k_rope_shared.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        k_rope_shared = self.rope(k_rope_shared)
        k = torch.cat([k_nope, k_rope_shared], dim=-1)  # [B, H, S, head_dim + rope_dim]

        # Pad V to match the effective Q/K dimension for flash attention
        v_padded = torch.nn.functional.pad(v, (0, self.rope_dim))

        # Flash attention
        if self.training:
            attn_out = fastest_flash_attn_func(
                q, k, v_padded,
                scale=self.scale,
                causal=use_causal,
                block_size=bs or 64,
            )
        else:
            attn_out, _ = fastest_flash_attn_forward(
                q, k, v_padded,
                scale=self.scale,
                causal=use_causal,
                block_size=bs,
            )

        # Reshape and project
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(attn_out)

    def extra_repr(self) -> str:
        return (
            f"hidden={self.hidden_size}, heads={self.num_heads}, "
            f"head_dim={self.head_dim}, latent={self.latent_dim}, "
            f"rope_dim={self.rope_dim}"
        )
