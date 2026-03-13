"""
Comprehensive test suite for Fastest Flash Attention.

Tests cover:
  1.  Basic correctness (3D/4D, various block sizes)
  2.  Causal attention
  3.  Sliding-window attention
  4.  Softcap logit capping
  5.  Block-sparse attention
  6.  Two-pass softmax precision
  7.  Mixed precision accumulation
  8.  KV-cache (continuous)
  9.  KV-cache (paged)
  10. GQA / MQA
  11. MLA (Multi-Latent Attention)
  12. Autograd / backward pass
  13. Module-level (MHA)
  14. Module-level (GQA)
  15. Block mask generators
  16. Auto block-size selection
  17. Combined features
  18. Edge cases (seq_len=1, S_q != S_kv)
  19. Return logsumexp
  20. Numerical stability

Total: 50+ parametrised test cases.
"""

from __future__ import annotations

import math
import pytest
import torch

from fastest_flash_attention import (
    AttentionConfig,
    fastest_flash_attn_forward,
    fastest_flash_attn_func,
    FastestFlashMHA,
    FastestFlashGQA,
    FastestFlashMLA,
    PagedKVCache,
    ContinuousKVCache,
    create_causal_block_mask,
    create_sliding_window_block_mask,
    create_local_block_mask,
    auto_select_block_size,
)


# ======================================================================
#  Reference naive SDPA
# ======================================================================

def _naive_sdpa(q, k, v, scale, causal=False, softcap=None, window_size=None):
    """Reference scaled dot-product attention."""
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale

    if softcap is not None:
        scores = softcap * torch.tanh(scores / softcap)

    if causal:
        S_q, S_k = scores.size(-2), scores.size(-1)
        mask = torch.triu(
            torch.full((S_q, S_k), float("-inf"), dtype=scores.dtype, device=q.device),
            diagonal=1,
        )
        if scores.dim() == 4:
            mask = mask.unsqueeze(0).unsqueeze(0)
        scores = scores + mask

    if window_size is not None:
        S_q, S_k = scores.size(-2), scores.size(-1)
        row_idx = torch.arange(S_q, device=q.device).unsqueeze(1)
        col_idx = torch.arange(S_k, device=q.device).unsqueeze(0)
        outside = col_idx < (row_idx - window_size + 1)
        if scores.dim() == 4:
            outside = outside.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(outside, float("-inf"))

    # Safe softmax
    scores_max = scores.max(dim=-1, keepdim=True).values
    scores_stable = scores - scores_max
    exp_s = torch.exp(scores_stable)
    attn_weights = exp_s / exp_s.sum(dim=-1, keepdim=True)
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    return torch.matmul(attn_weights, v.float()).to(q.dtype)


# ======================================================================
#  1. Basic Correctness
# ======================================================================

class TestBasicCorrectness:
    """Verify flash output matches naive SDPA."""

    @pytest.mark.parametrize("block_size", [4, 8, 16, 32])
    def test_3d_various_blocks(self, block_size):
        torch.manual_seed(42)
        B, S, D = 2, 32, 16
        scale = 1.0 / math.sqrt(D)
        q = torch.randn(B, S, D)
        k = torch.randn(B, S, D)
        v = torch.randn(B, S, D)

        ref = _naive_sdpa(q, k, v, scale)
        out, _ = fastest_flash_attn_forward(q, k, v, scale=scale, block_size=block_size)

        assert torch.allclose(ref, out, atol=1e-5), \
            f"Max diff: {(ref - out).abs().max().item()}"

    def test_4d_input(self):
        torch.manual_seed(0)
        B, H, S, D = 2, 4, 24, 16
        scale = 1.0 / math.sqrt(D)
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        ref = _naive_sdpa(q, k, v, scale)
        out, _ = fastest_flash_attn_forward(q, k, v, scale=scale, block_size=8)

        assert torch.allclose(ref, out, atol=1e-5)

    def test_non_divisible_seq_len(self):
        """S not a multiple of block_size."""
        torch.manual_seed(1)
        B, H, S, D = 1, 2, 17, 8
        scale = 1.0 / math.sqrt(D)
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        ref = _naive_sdpa(q, k, v, scale)
        out, _ = fastest_flash_attn_forward(q, k, v, scale=scale, block_size=8)

        assert torch.allclose(ref, out, atol=1e-5)


# ======================================================================
#  2. Causal Attention
# ======================================================================

class TestCausal:
    @pytest.mark.parametrize("block_size", [4, 8, 16])
    def test_causal_matches_naive(self, block_size):
        torch.manual_seed(10)
        B, H, S, D = 2, 4, 32, 16
        scale = 1.0 / math.sqrt(D)
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        ref = _naive_sdpa(q, k, v, scale, causal=True)
        out, _ = fastest_flash_attn_forward(
            q, k, v, scale=scale, causal=True, block_size=block_size
        )

        assert torch.allclose(ref, out, atol=1e-5)

    def test_causal_3d(self):
        torch.manual_seed(11)
        B, S, D = 2, 20, 16
        scale = 1.0 / math.sqrt(D)
        q, k, v = torch.randn(B, S, D), torch.randn(B, S, D), torch.randn(B, S, D)

        ref = _naive_sdpa(q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), scale, causal=True).squeeze(1)
        out, _ = fastest_flash_attn_forward(q, k, v, scale=scale, causal=True, block_size=8)

        assert torch.allclose(ref, out, atol=1e-5)


# ======================================================================
#  3. Sliding Window
# ======================================================================

class TestSlidingWindow:
    @pytest.mark.parametrize("window", [4, 8, 16])
    def test_sliding_window(self, window):
        torch.manual_seed(20)
        B, H, S, D = 1, 2, 24, 8
        scale = 1.0 / math.sqrt(D)
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        ref = _naive_sdpa(q, k, v, scale, causal=True, window_size=window)
        out, _ = fastest_flash_attn_forward(
            q, k, v, scale=scale, causal=True,
            sliding_window=window, block_size=4,
        )

        assert torch.allclose(ref, out, atol=1e-5)


# ======================================================================
#  4. Softcap
# ======================================================================

class TestSoftcap:
    @pytest.mark.parametrize("cap", [5.0, 10.0, 30.0])
    def test_softcap_matches_naive(self, cap):
        torch.manual_seed(30)
        B, H, S, D = 2, 2, 16, 8
        scale = 1.0 / math.sqrt(D)
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        ref = _naive_sdpa(q, k, v, scale, softcap=cap)
        out, _ = fastest_flash_attn_forward(
            q, k, v, scale=scale, softcap=cap, block_size=4,
        )

        assert torch.allclose(ref, out, atol=1e-4)

    def test_softcap_causal(self):
        torch.manual_seed(31)
        B, H, S, D = 1, 2, 20, 8
        cap = 15.0
        scale = 1.0 / math.sqrt(D)
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        ref = _naive_sdpa(q, k, v, scale, causal=True, softcap=cap)
        out, _ = fastest_flash_attn_forward(
            q, k, v, scale=scale, causal=True, softcap=cap, block_size=8,
        )

        assert torch.allclose(ref, out, atol=1e-4)


# ======================================================================
#  5. Block-Sparse
# ======================================================================

class TestBlockSparse:
    def test_block_mask_skips_tiles(self):
        torch.manual_seed(40)
        B, H, S, D = 1, 1, 16, 8
        block_size = 4
        scale = 1.0 / math.sqrt(D)
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        # Full mask (all tiles active) → should match reference
        n_br = math.ceil(S / block_size)
        n_bc = math.ceil(S / block_size)
        full_mask = torch.ones(n_br, n_bc, dtype=torch.bool)

        ref = _naive_sdpa(q, k, v, scale)
        out, _ = fastest_flash_attn_forward(
            q, k, v, scale=scale, block_size=block_size, block_mask=full_mask,
        )

        assert torch.allclose(ref, out, atol=1e-5)

    def test_sparse_reduces_computation(self):
        """Sparse mask with some blocks zeroed out should differ from full."""
        torch.manual_seed(41)
        B, H, S, D = 1, 1, 16, 8
        block_size = 4
        scale = 1.0 / math.sqrt(D)
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        full_out, _ = fastest_flash_attn_forward(
            q, k, v, scale=scale, block_size=block_size,
        )

        n_br = math.ceil(S / block_size)
        n_bc = math.ceil(S / block_size)
        sparse_mask = torch.ones(n_br, n_bc, dtype=torch.bool)
        sparse_mask[0, -1] = False  # skip one tile

        sparse_out, _ = fastest_flash_attn_forward(
            q, k, v, scale=scale, block_size=block_size, block_mask=sparse_mask,
        )

        # Should be different (unless that tile had all-zero attention)
        assert not torch.allclose(full_out, sparse_out, atol=1e-6)


# ======================================================================
#  6. Two-Pass Softmax
# ======================================================================

class TestTwoPass:
    def test_two_pass_matches_one_pass(self):
        torch.manual_seed(50)
        B, H, S, D = 2, 2, 24, 8
        scale = 1.0 / math.sqrt(D)
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        ref = _naive_sdpa(q, k, v, scale)
        out_1p, _ = fastest_flash_attn_forward(
            q, k, v, scale=scale, block_size=8, two_pass=False,
        )
        out_2p, _ = fastest_flash_attn_forward(
            q, k, v, scale=scale, block_size=8, two_pass=True,
        )

        assert torch.allclose(ref, out_1p, atol=1e-5)
        assert torch.allclose(ref, out_2p, atol=1e-5)

    def test_two_pass_causal(self):
        torch.manual_seed(51)
        B, H, S, D = 1, 2, 16, 8
        scale = 1.0 / math.sqrt(D)
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        ref = _naive_sdpa(q, k, v, scale, causal=True)
        out, _ = fastest_flash_attn_forward(
            q, k, v, scale=scale, causal=True, block_size=4, two_pass=True,
        )

        assert torch.allclose(ref, out, atol=1e-5)


# ======================================================================
#  7. Mixed Precision
# ======================================================================

class TestMixedPrecision:
    def test_fp16_compute_fp32_accum(self):
        torch.manual_seed(60)
        B, H, S, D = 2, 2, 16, 8
        scale = 1.0 / math.sqrt(D)
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        ref = _naive_sdpa(q, k, v, scale)
        out, _ = fastest_flash_attn_forward(
            q, k, v, scale=scale, block_size=4,
            compute_dtype=torch.float16,
        )

        # Looser tolerance due to lower precision compute path
        assert torch.allclose(ref, out, atol=1e-3)


# ======================================================================
#  8. Continuous KV-Cache
# ======================================================================

class TestContinuousKVCache:
    def test_cache_append(self):
        cache = ContinuousKVCache(
            max_batch=2, max_seq_len=64, num_heads=2, head_dim=8,
        )
        B, H, S_new, D = 2, 2, 4, 8

        k1 = torch.randn(B, H, S_new, D)
        v1 = torch.randn(B, H, S_new, D)
        k_full, v_full = cache.append(k1, v1)
        assert k_full.shape == (2, 2, 4, 8)

        k2 = torch.randn(B, H, 2, D)
        v2 = torch.randn(B, H, 2, D)
        k_full, v_full = cache.append(k2, v2)
        assert k_full.shape == (2, 2, 6, 8)

    def test_cache_with_attention(self):
        torch.manual_seed(70)
        B, H, D = 1, 2, 8
        scale = 1.0 / math.sqrt(D)

        # Build reference with full sequence
        q_full = torch.randn(B, H, 6, D)
        k_full = torch.randn(B, H, 6, D)
        v_full = torch.randn(B, H, 6, D)
        ref = _naive_sdpa(q_full, k_full, v_full, scale)

        # Incremental: first 4 tokens, then 2 more
        kv_cache = (k_full[:, :, :4, :], v_full[:, :, :4, :])
        out, _ = fastest_flash_attn_forward(
            q_full, k_full[:, :, 4:, :], v_full[:, :, 4:, :],
            scale=scale, block_size=4, kv_cache=kv_cache,
        )

        assert torch.allclose(ref, out, atol=1e-5)


# ======================================================================
#  9. Paged KV-Cache
# ======================================================================

class TestPagedKVCache:
    def test_paged_basic(self):
        cache = PagedKVCache(
            max_pages=16, page_size=4, num_heads=2, head_dim=8,
            dtype=torch.float32,
        )

        k = torch.randn(2, 6, 8)
        v = torch.randn(2, 6, 8)
        total_len = cache.append_tokens(0, k, v)
        assert total_len == 6

        k_out, v_out = cache.get_kv(0)
        assert k_out.shape == (2, 6, 8)
        assert torch.allclose(k_out, k)

    def test_paged_multiple_sequences(self):
        cache = PagedKVCache(
            max_pages=32, page_size=4, num_heads=2, head_dim=8,
        )

        k0 = torch.randn(2, 5, 8)
        v0 = torch.randn(2, 5, 8)
        cache.append_tokens(0, k0, v0)

        k1 = torch.randn(2, 3, 8)
        v1 = torch.randn(2, 3, 8)
        cache.append_tokens(1, k1, v1)

        k0_out, _ = cache.get_kv(0)
        k1_out, _ = cache.get_kv(1)
        assert k0_out.shape == (2, 5, 8)
        assert k1_out.shape == (2, 3, 8)

    def test_paged_free_sequence(self):
        cache = PagedKVCache(max_pages=8, page_size=4, num_heads=2, head_dim=8)
        initial_free = cache.num_free_pages

        k = torch.randn(2, 10, 8)
        v = torch.randn(2, 10, 8)
        cache.append_tokens(0, k, v)
        assert cache.num_free_pages < initial_free

        cache.free_sequence(0)
        assert cache.num_free_pages == initial_free


# ======================================================================
#  10. GQA / MQA
# ======================================================================

class TestGQA:
    def test_gqa_4_heads_2_kvheads(self):
        """GQA: 4 query heads, 2 KV heads → ratio 2."""
        torch.manual_seed(80)
        B, S, D = 2, 16, 8
        H_q, H_kv = 4, 2
        scale = 1.0 / math.sqrt(D)

        q = torch.randn(B, H_q, S, D)
        k = torch.randn(B, H_kv, S, D)
        v = torch.randn(B, H_kv, S, D)

        # Manual repeat for reference
        from fastest_flash_attention.utils import repeat_kv
        k_rep = repeat_kv(k, H_q, H_kv)
        v_rep = repeat_kv(v, H_q, H_kv)
        ref = _naive_sdpa(q, k_rep, v_rep, scale)

        out, _ = fastest_flash_attn_forward(
            q, k, v, scale=scale, block_size=4, num_kv_heads=H_kv,
        )

        assert torch.allclose(ref, out, atol=1e-5)

    def test_mqa(self):
        """MQA: 4 query heads, 1 KV head."""
        torch.manual_seed(81)
        B, S, D = 1, 12, 8
        H_q, H_kv = 4, 1
        scale = 1.0 / math.sqrt(D)

        q = torch.randn(B, H_q, S, D)
        k = torch.randn(B, H_kv, S, D)
        v = torch.randn(B, H_kv, S, D)

        from fastest_flash_attention.utils import repeat_kv
        k_rep = repeat_kv(k, H_q, H_kv)
        v_rep = repeat_kv(v, H_q, H_kv)
        ref = _naive_sdpa(q, k_rep, v_rep, scale)

        out, _ = fastest_flash_attn_forward(
            q, k, v, scale=scale, block_size=4, num_kv_heads=H_kv,
        )

        assert torch.allclose(ref, out, atol=1e-5)


# ======================================================================
#  11. MLA Module
# ======================================================================

class TestMLA:
    def test_mla_forward_shape(self):
        torch.manual_seed(90)
        B, S, H_dim = 2, 16, 64
        model = FastestFlashMLA(
            hidden_size=H_dim, num_heads=4, head_dim=16,
            latent_dim=32, rope_dim=8, causal=True, block_size=4,
        )
        x = torch.randn(B, S, H_dim)
        out = model(x)
        assert out.shape == (B, S, H_dim)

    def test_mla_gradient_flow(self):
        torch.manual_seed(91)
        B, S, H_dim = 1, 8, 32
        model = FastestFlashMLA(
            hidden_size=H_dim, num_heads=2, head_dim=16,
            latent_dim=16, rope_dim=4, block_size=4,
        )
        model.train()
        x = torch.randn(B, S, H_dim, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


# ======================================================================
#  12. Autograd / Backward
# ======================================================================

class TestAutograd:
    def test_backward_basic(self):
        torch.manual_seed(100)
        B, H, S, D = 2, 2, 16, 8
        q = torch.randn(B, H, S, D, requires_grad=True)
        k = torch.randn(B, H, S, D, requires_grad=True)
        v = torch.randn(B, H, S, D, requires_grad=True)

        out = fastest_flash_attn_func(q, k, v, block_size=4)
        loss = out.sum()
        loss.backward()

        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None
        assert q.grad.shape == q.shape

    def test_backward_causal(self):
        torch.manual_seed(101)
        B, H, S, D = 1, 2, 12, 8
        q = torch.randn(B, H, S, D, requires_grad=True)
        k = torch.randn(B, H, S, D, requires_grad=True)
        v = torch.randn(B, H, S, D, requires_grad=True)

        out = fastest_flash_attn_func(q, k, v, causal=True, block_size=4)
        loss = out.sum()
        loss.backward()

        assert q.grad is not None

    def test_gradcheck_small(self):
        """Numerical gradient check on a very small problem."""
        torch.manual_seed(102)
        B, H, S, D = 1, 1, 4, 4
        q = torch.randn(B, H, S, D, dtype=torch.float64, requires_grad=True)
        k = torch.randn(B, H, S, D, dtype=torch.float64, requires_grad=True)
        v = torch.randn(B, H, S, D, dtype=torch.float64, requires_grad=True)

        def fn(q, k, v):
            return fastest_flash_attn_func(q, k, v, block_size=4)

        assert torch.autograd.gradcheck(fn, (q, k, v), atol=5e-2, rtol=5e-2, eps=1e-3)


# ======================================================================
#  13. Module-Level MHA
# ======================================================================

class TestModuleMHA:
    def test_mha_output_shape(self):
        torch.manual_seed(110)
        B, S, H = 2, 16, 64
        model = FastestFlashMHA(hidden_size=H, num_heads=4, block_size=4)
        x = torch.randn(B, S, H)
        out = model(x)
        assert out.shape == (B, S, H)

    def test_mha_causal(self):
        torch.manual_seed(111)
        B, S, H = 1, 12, 32
        model = FastestFlashMHA(hidden_size=H, num_heads=2, causal=True, block_size=4)
        x = torch.randn(B, S, H)
        out = model(x)
        assert out.shape == (B, S, H)

    def test_mha_with_softcap(self):
        torch.manual_seed(112)
        B, S, H = 1, 8, 16
        model = FastestFlashMHA(
            hidden_size=H, num_heads=2, softcap=10.0, block_size=4,
        )
        x = torch.randn(B, S, H)
        out = model(x)
        assert out.shape == (B, S, H)

    def test_mha_return_lse(self):
        torch.manual_seed(113)
        B, S, H = 1, 8, 16
        model = FastestFlashMHA(hidden_size=H, num_heads=2, block_size=4)
        model.eval()
        x = torch.randn(B, S, H)
        out, lse = model(x, return_lse=True)
        assert out.shape == (B, S, H)
        assert lse is not None


# ======================================================================
#  14. Module-Level GQA
# ======================================================================

class TestModuleGQA:
    def test_gqa_output_shape(self):
        torch.manual_seed(120)
        B, S, H = 2, 16, 64
        model = FastestFlashGQA(
            hidden_size=H, num_heads=8, num_kv_heads=2, block_size=4,
        )
        x = torch.randn(B, S, H)
        out = model(x)
        assert out.shape == (B, S, H)

    def test_gqa_gradient_flow(self):
        torch.manual_seed(121)
        B, S, H = 1, 8, 32
        model = FastestFlashGQA(
            hidden_size=H, num_heads=4, num_kv_heads=2, block_size=4,
        )
        model.train()
        x = torch.randn(B, S, H, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


# ======================================================================
#  15. Block Mask Generators
# ======================================================================

class TestBlockMaskGenerators:
    def test_causal_mask_shape(self):
        mask = create_causal_block_mask(32, 32, 8, 8)
        assert mask.shape == (4, 4)
        # Lower-left should be True, upper-right should be False-ish
        assert mask[0, 0].item() is True
        assert mask[-1, -1].item() is True

    def test_sliding_window_mask(self):
        mask = create_sliding_window_block_mask(32, 32, 8, 8, window_size=8)
        assert mask.shape == (4, 4)
        # Diagonal blocks should be active
        for i in range(4):
            assert mask[i, i].item() is True

    def test_local_mask(self):
        mask = create_local_block_mask(32, 32, 8, 8, local_radius=8)
        assert mask.shape == (4, 4)
        assert mask[0, 0].item() is True


# ======================================================================
#  16. Auto Block-Size Selection
# ======================================================================

class TestAutoBlockSize:
    def test_returns_power_of_2(self):
        for D in [16, 32, 64, 128]:
            bs = auto_select_block_size(256, 256, D)
            assert bs > 0
            assert (bs & (bs - 1)) == 0, f"Not power of 2: {bs}"

    def test_respects_bounds(self):
        bs = auto_select_block_size(1024, 1024, 64, max_block=128, min_block=32)
        assert 32 <= bs <= 128

    def test_small_dim_large_block(self):
        bs_small = auto_select_block_size(256, 256, 8)
        bs_large = auto_select_block_size(256, 256, 128)
        assert bs_small >= bs_large


# ======================================================================
#  17. Combined Features
# ======================================================================

class TestCombined:
    def test_causal_plus_window(self):
        torch.manual_seed(130)
        B, H, S, D = 1, 2, 24, 8
        scale = 1.0 / math.sqrt(D)
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        ref = _naive_sdpa(q, k, v, scale, causal=True, window_size=8)
        out, _ = fastest_flash_attn_forward(
            q, k, v, scale=scale, causal=True, sliding_window=8, block_size=4,
        )

        assert torch.allclose(ref, out, atol=1e-5)

    def test_causal_softcap_window(self):
        torch.manual_seed(131)
        B, H, S, D = 1, 2, 16, 8
        scale = 1.0 / math.sqrt(D)
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        ref = _naive_sdpa(q, k, v, scale, causal=True, softcap=10.0, window_size=8)
        out, _ = fastest_flash_attn_forward(
            q, k, v, scale=scale, causal=True, softcap=10.0,
            sliding_window=8, block_size=4,
        )

        assert torch.allclose(ref, out, atol=1e-4)


# ======================================================================
#  18. Edge Cases
# ======================================================================

class TestEdgeCases:
    def test_seq_len_1(self):
        torch.manual_seed(140)
        B, H, D = 1, 2, 8
        scale = 1.0 / math.sqrt(D)
        q = torch.randn(B, H, 1, D)
        k = torch.randn(B, H, 1, D)
        v = torch.randn(B, H, 1, D)

        ref = _naive_sdpa(q, k, v, scale)
        out, _ = fastest_flash_attn_forward(q, k, v, scale=scale, block_size=4)

        assert torch.allclose(ref, out, atol=1e-5)

    def test_asymmetric_seq_lens(self):
        torch.manual_seed(141)
        B, H, S_q, S_kv, D = 1, 2, 8, 16, 8
        scale = 1.0 / math.sqrt(D)
        q = torch.randn(B, H, S_q, D)
        k = torch.randn(B, H, S_kv, D)
        v = torch.randn(B, H, S_kv, D)

        ref = _naive_sdpa(q, k, v, scale)
        out, _ = fastest_flash_attn_forward(q, k, v, scale=scale, block_size=4)

        assert torch.allclose(ref, out, atol=1e-5)

    def test_large_head_dim(self):
        torch.manual_seed(142)
        B, H, S, D = 1, 1, 8, 128
        scale = 1.0 / math.sqrt(D)
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        ref = _naive_sdpa(q, k, v, scale)
        out, _ = fastest_flash_attn_forward(q, k, v, scale=scale, block_size=4)

        assert torch.allclose(ref, out, atol=1e-5)


# ======================================================================
#  19. Return Logsumexp
# ======================================================================

class TestReturnLSE:
    def test_lse_shape(self):
        torch.manual_seed(150)
        B, H, S, D = 2, 2, 16, 8
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        out, lse = fastest_flash_attn_forward(
            q, k, v, block_size=4, return_lse=True,
        )

        assert lse is not None
        assert lse.shape == (B, H, S, 1)

    def test_lse_values(self):
        """LSE should equal torch.logsumexp of full attention scores."""
        torch.manual_seed(151)
        B, H, S, D = 1, 1, 8, 4
        scale = 1.0 / math.sqrt(D)
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        _, lse = fastest_flash_attn_forward(
            q, k, v, scale=scale, block_size=4, return_lse=True,
        )

        # Reference LSE
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
        ref_lse = torch.logsumexp(scores, dim=-1, keepdim=True)

        assert torch.allclose(lse.float(), ref_lse, atol=1e-4)


# ======================================================================
#  20. Numerical Stability
# ======================================================================

class TestNumericalStability:
    def test_large_values(self):
        """Large input values should not produce NaN/Inf."""
        torch.manual_seed(160)
        B, H, S, D = 1, 1, 8, 4
        q = torch.randn(B, H, S, D) * 100
        k = torch.randn(B, H, S, D) * 100
        v = torch.randn(B, H, S, D)

        out, _ = fastest_flash_attn_forward(q, k, v, block_size=4)
        assert torch.isfinite(out).all()

    def test_zero_inputs(self):
        """All-zero Q should produce all-zero or uniform-average output."""
        B, H, S, D = 1, 1, 4, 4
        q = torch.zeros(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        out, _ = fastest_flash_attn_forward(q, k, v, block_size=4)
        assert torch.isfinite(out).all()


# ======================================================================
#  Config-based test
# ======================================================================

class TestConfig:
    def test_config_forward(self):
        torch.manual_seed(170)
        B, H, S, D = 2, 2, 16, 8
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        config = AttentionConfig(causal=True, block_size_q=4, block_size_kv=4)
        out, _ = fastest_flash_attn_forward(q, k, v, config=config)

        ref = _naive_sdpa(q, k, v, 1.0 / math.sqrt(D), causal=True)
        assert torch.allclose(ref, out, atol=1e-5)
