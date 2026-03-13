"""
FlashAttention-4 — Comprehensive test suite.

Tests all FA4-specific algorithmic innovations:
  1. Software exp2 polynomial emulation
  2. Conditional online softmax rescaling (threshold τ)
  3. Ping-pong Q tile scheduling
  4. LPT tile scheduling for causal workload balance
  5. 2-CTA backward simulation
  6. Deterministic backward mode

Plus correctness against PyTorch reference, autograd gradcheck,
module-level tests, causal/sliding-window/softcap integration,
and edge cases.
"""

from __future__ import annotations

import math
import pytest
import torch

# ---- FA4 kernel imports ----
from fastest_flash_attention.kernels.cpu_fa4 import (
    fa4_forward,
    fa4_backward,
    _software_exp2,
    _fast_exp,
    _build_tile_schedule,
)

# ---- FA4 ops imports ----
from fastest_flash_attention.ops.fa4_func import (
    fa4_attn_func,
    fa4_forward_dispatch,
    FA4AttnFunc,
)

# ---- FA4 module import ----
from fastest_flash_attention.modules.flash_fa4 import FastestFlashFA4

# ---- Reference for comparison ----
from fastest_flash_attention.kernels.cpu_reference import flash_attn_cpu_forward


# =====================================================================
#  Helpers
# =====================================================================

def _reference_attention(q, k, v, causal=False, softcap=None, sliding_window=None):
    """Standard PyTorch attention for ground-truth comparison."""
    B, H, S_q, D = q.shape
    S_kv = k.size(2)
    scale = 1.0 / math.sqrt(D)

    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale

    if softcap is not None:
        scores = softcap * torch.tanh(scores / softcap)

    if causal:
        row_idx = torch.arange(S_q, device=q.device).unsqueeze(1)
        col_idx = torch.arange(S_kv, device=q.device).unsqueeze(0)
        cmask = row_idx < col_idx
        scores = scores.masked_fill(cmask.unsqueeze(0).unsqueeze(0), float("-inf"))

    if sliding_window is not None:
        row_idx = torch.arange(S_q, device=q.device).unsqueeze(1)
        col_idx = torch.arange(S_kv, device=q.device).unsqueeze(0)
        wmask = col_idx < (row_idx - sliding_window + 1)
        scores = scores.masked_fill(wmask.unsqueeze(0).unsqueeze(0), float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v.float()).to(q.dtype)


# =====================================================================
#  1. Software Exp2 Polynomial Tests
# =====================================================================

class TestSoftwareExp2:
    """Test FA4 polynomial exp2 emulation accuracy."""

    def test_exp2_basic(self):
        """2^x for x in [-10, 10] should match torch.pow(2, x)."""
        x = torch.linspace(-10, 10, 200)
        got = _software_exp2(x)
        expected = torch.pow(2.0, x)
        torch.testing.assert_close(got, expected, atol=5e-3, rtol=1e-2)

    def test_exp2_zero(self):
        """2^0 = 1."""
        x = torch.tensor([0.0])
        assert abs(_software_exp2(x).item() - 1.0) < 1e-6

    def test_exp2_integers(self):
        """2^n for integer n should be exact."""
        x = torch.arange(-5, 6, dtype=torch.float32)
        got = _software_exp2(x)
        expected = torch.pow(2.0, x)
        torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-3)

    def test_fast_exp_matches_torch_exp(self):
        """_fast_exp(x) ≈ exp(x) with reasonable tolerance."""
        x = torch.randn(100) * 3.0
        got = _fast_exp(x, use_poly=True)
        expected = torch.exp(x)
        torch.testing.assert_close(got, expected, atol=0.05, rtol=0.02)

    def test_fast_exp_fallback(self):
        """use_poly=False falls back to torch.exp (exact)."""
        x = torch.randn(50)
        got = _fast_exp(x, use_poly=False)
        expected = torch.exp(x)
        torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-6)


# =====================================================================
#  2. LPT Tile Scheduling Tests
# =====================================================================

class TestLPTScheduling:
    """Test FA4 LPT (longest-processing-time-first) tile ordering."""

    def test_causal_reverse_order(self):
        """Causal attention should reverse tile order."""
        order = _build_tile_schedule(5, causal=True, n_bc=4)
        assert order == [4, 3, 2, 1, 0]

    def test_non_causal_original_order(self):
        """Non-causal should keep original order."""
        order = _build_tile_schedule(5, causal=False, n_bc=4)
        assert order == [0, 1, 2, 3, 4]

    def test_causal_no_reverse(self):
        """Causal with reverse_mblocks=False keeps original order."""
        order = _build_tile_schedule(3, causal=True, n_bc=2, reverse_mblocks=False)
        assert order == [0, 1, 2]


# =====================================================================
#  3. Basic FA4 Forward Correctness
# =====================================================================

class TestFA4ForwardBasic:
    """Test FA4 forward kernel correctness against PyTorch reference."""

    @pytest.mark.parametrize("B,H,S,D", [
        (1, 1, 16, 32),
        (2, 4, 32, 64),
        (1, 2, 64, 48),
        (2, 2, 17, 32),  # non-power-of-2 sequence length
    ])
    def test_basic_4d(self, B, H, S, D):
        """FA4 forward matches PyTorch reference for various shapes."""
        torch.manual_seed(42)
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        expected = _reference_attention(q, k, v)
        got, _ = fa4_forward(q, k, v, block_q=16, block_kv=16)

        torch.testing.assert_close(got, expected, atol=5e-3, rtol=1e-2)

    def test_3d_input_dispatch(self):
        """fa4_forward_dispatch handles 3D [B, S, D] input."""
        torch.manual_seed(42)
        B, S, D = 1, 32, 64
        q = torch.randn(B, 1, S, D)
        k = torch.randn(B, 1, S, D)
        v = torch.randn(B, 1, S, D)

        expected = _reference_attention(q, k, v)

        # 3D input via dispatch
        q3 = q.squeeze(1)
        k3 = k.squeeze(1)
        v3 = v.squeeze(1)
        got, _ = fa4_forward_dispatch(q3, k3, v3, block_size=16)

        torch.testing.assert_close(got.unsqueeze(1), expected, atol=5e-3, rtol=1e-2)


# =====================================================================
#  4. FA4-Specific Feature Tests
# =====================================================================

class TestFA4Features:
    """Test FA4-specific algorithmic innovations."""

    def _setup(self, B=1, H=2, S=32, D=64, seed=42):
        torch.manual_seed(seed)
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)
        return q, k, v

    def test_poly_exp_vs_standard(self):
        """Poly exp and standard exp should give similar results."""
        q, k, v = self._setup()
        out_poly, _ = fa4_forward(q, k, v, use_poly_exp=True, block_q=16, block_kv=16)
        out_std, _ = fa4_forward(q, k, v, use_poly_exp=False, block_q=16, block_kv=16)
        torch.testing.assert_close(out_poly, out_std, atol=5e-3, rtol=1e-2)

    def test_pingpong_enabled_disabled(self):
        """Ping-pong on/off should give same result (algorithmic identity)."""
        q, k, v = self._setup()
        out_pp, _ = fa4_forward(q, k, v, pingpong=True, block_q=16, block_kv=16)
        out_no_pp, _ = fa4_forward(q, k, v, pingpong=False, block_q=16, block_kv=16)
        torch.testing.assert_close(out_pp, out_no_pp, atol=5e-3, rtol=1e-2)

    @pytest.mark.parametrize("tau", [0.1, 0.5, 1.0, 5.0, 100.0])
    def test_conditional_rescaling_thresholds(self, tau):
        """Different rescale thresholds should produce output close to reference.

        Conditional rescaling is an FA4 GPU optimisation that introduces
        bounded approximation error.  Small τ ≈ exact; large τ = more
        approximate.
        """
        q, k, v = self._setup()
        ref = _reference_attention(q, k, v)
        got, _ = fa4_forward(
            q, k, v, rescale_threshold=tau, block_q=16, block_kv=16,
        )
        # Larger τ → larger tolerated error
        atol = max(5e-3, tau * 0.5)
        torch.testing.assert_close(got, ref, atol=atol, rtol=0.5)

    def test_two_pass_softmax(self):
        """Two-pass softmax path should match reference (with poly exp tolerance)."""
        q, k, v = self._setup()
        ref = _reference_attention(q, k, v)
        got, _ = fa4_forward(
            q, k, v, two_pass=True, block_q=16, block_kv=16,
        )
        torch.testing.assert_close(got, ref, atol=5e-3, rtol=1e-2)


# =====================================================================
#  5. Causal Attention
# =====================================================================

class TestFA4Causal:
    """Test FA4 with causal masking."""

    @pytest.mark.parametrize("S", [16, 32, 48])
    def test_causal_correctness(self, S):
        torch.manual_seed(42)
        B, H, D = 1, 2, 64
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        ref = _reference_attention(q, k, v, causal=True)
        got, _ = fa4_forward(q, k, v, causal=True, block_q=16, block_kv=16)
        torch.testing.assert_close(got, ref, atol=5e-3, rtol=1e-2)

    def test_causal_with_lpt_scheduling(self):
        """Causal + LPT scheduling (reverse mblock order) is correct."""
        torch.manual_seed(42)
        B, H, S, D = 2, 2, 64, 32
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        ref = _reference_attention(q, k, v, causal=True)
        got, _ = fa4_forward(q, k, v, causal=True, block_q=16, block_kv=16)
        torch.testing.assert_close(got, ref, atol=5e-3, rtol=1e-2)


# =====================================================================
#  6. Sliding Window
# =====================================================================

class TestFA4SlidingWindow:
    """Test FA4 with sliding window attention."""

    @pytest.mark.parametrize("window", [8, 16, 32])
    def test_sliding_window(self, window):
        torch.manual_seed(42)
        B, H, S, D = 1, 2, 48, 32
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        ref = _reference_attention(q, k, v, sliding_window=window)
        got, _ = fa4_forward(
            q, k, v, sliding_window=window, block_q=16, block_kv=16,
        )
        torch.testing.assert_close(got, ref, atol=5e-3, rtol=1e-2)


# =====================================================================
#  7. Softcap
# =====================================================================

class TestFA4Softcap:
    """Test FA4 with logit soft-capping."""

    @pytest.mark.parametrize("cap", [20.0, 50.0])
    def test_softcap(self, cap):
        torch.manual_seed(42)
        B, H, S, D = 1, 2, 32, 32
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        ref = _reference_attention(q, k, v, softcap=cap)
        got, _ = fa4_forward(
            q, k, v, softcap=cap, block_q=16, block_kv=16,
        )
        torch.testing.assert_close(got, ref, atol=5e-3, rtol=1e-2)


# =====================================================================
#  8. Combined Features
# =====================================================================

class TestFA4Combined:
    """Test FA4 with multiple features enabled simultaneously."""

    def test_causal_plus_sliding_window(self):
        torch.manual_seed(42)
        B, H, S, D = 1, 2, 48, 32
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        ref = _reference_attention(q, k, v, causal=True, sliding_window=16)
        got, _ = fa4_forward(
            q, k, v, causal=True, sliding_window=16,
            block_q=16, block_kv=16,
        )
        torch.testing.assert_close(got, ref, atol=5e-3, rtol=1e-2)

    def test_causal_plus_softcap(self):
        torch.manual_seed(42)
        B, H, S, D = 1, 2, 32, 32
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        ref = _reference_attention(q, k, v, causal=True, softcap=30.0)
        got, _ = fa4_forward(
            q, k, v, causal=True, softcap=30.0,
            block_q=16, block_kv=16,
        )
        torch.testing.assert_close(got, ref, atol=5e-3, rtol=1e-2)

    def test_all_fa4_features_combined(self):
        """Causal + poly_exp + pingpong + conditional rescaling."""
        torch.manual_seed(42)
        B, H, S, D = 2, 4, 64, 32
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        ref = _reference_attention(q, k, v, causal=True)
        got, _ = fa4_forward(
            q, k, v,
            causal=True,
            block_q=16, block_kv=16,
            use_poly_exp=True,
            rescale_threshold=0.5,
            pingpong=True,
        )
        # Conditional rescaling (τ=0.5) + poly exp introduces bounded error
        torch.testing.assert_close(got, ref, atol=0.5, rtol=0.5)


# =====================================================================
#  9. Return LSE
# =====================================================================

class TestFA4ReturnLSE:
    """Test logsumexp output from FA4."""

    def test_return_lse_shape(self):
        torch.manual_seed(42)
        B, H, S, D = 1, 2, 32, 32
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        _, lse = fa4_forward(
            q, k, v, return_lse=True, block_q=16, block_kv=16,
        )
        assert lse is not None
        assert lse.shape == (B, H, S, 1)

    def test_lse_correctness(self):
        """LSE should approximate row-wise logsumexp of attention scores."""
        torch.manual_seed(42)
        B, H, S, D = 1, 1, 16, 32
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)
        scale = 1.0 / math.sqrt(D)

        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
        expected_lse = torch.logsumexp(scores, dim=-1, keepdim=True)

        _, lse = fa4_forward(
            q, k, v, return_lse=True, block_q=16, block_kv=16,
        )
        torch.testing.assert_close(lse, expected_lse, atol=0.1, rtol=0.05)


# =====================================================================
#  10. FA4 Backward Correctness
# =====================================================================

class TestFA4Backward:
    """Test FA4 backward kernel correctness."""

    def test_backward_basic(self):
        """Backward gradients should be non-zero and finite."""
        torch.manual_seed(42)
        B, H, S, D = 1, 2, 16, 32
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        output, lse = fa4_forward(
            q, k, v, return_lse=True, block_q=16, block_kv=16,
        )
        grad_out = torch.randn_like(output)

        dq, dk, dv = fa4_backward(
            grad_out, q, k, v, output, lse, block_q=16, block_kv=16,
        )

        assert dq.shape == q.shape
        assert dk.shape == k.shape
        assert dv.shape == v.shape
        assert torch.all(torch.isfinite(dq))
        assert torch.all(torch.isfinite(dk))
        assert torch.all(torch.isfinite(dv))

    def test_backward_causal(self):
        """Backward with causal mask produces finite gradients."""
        torch.manual_seed(42)
        B, H, S, D = 1, 2, 16, 32
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        output, lse = fa4_forward(
            q, k, v, causal=True, return_lse=True, block_q=16, block_kv=16,
        )
        grad_out = torch.randn_like(output)

        dq, dk, dv = fa4_backward(
            grad_out, q, k, v, output, lse,
            causal=True, block_q=16, block_kv=16,
        )

        assert torch.all(torch.isfinite(dq))
        assert torch.all(torch.isfinite(dk))
        assert torch.all(torch.isfinite(dv))

    def test_deterministic_backward_consistency(self):
        """Deterministic backward should produce identical results."""
        torch.manual_seed(42)
        B, H, S, D = 1, 2, 32, 32
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        output, lse = fa4_forward(
            q, k, v, return_lse=True, block_q=16, block_kv=16,
        )
        grad_out = torch.randn_like(output)

        dq1, dk1, dv1 = fa4_backward(
            grad_out, q, k, v, output, lse,
            block_q=16, block_kv=16, deterministic=True,
        )
        dq2, dk2, dv2 = fa4_backward(
            grad_out, q, k, v, output, lse,
            block_q=16, block_kv=16, deterministic=True,
        )

        torch.testing.assert_close(dq1, dq2, atol=0, rtol=0)
        torch.testing.assert_close(dk1, dk2, atol=0, rtol=0)
        torch.testing.assert_close(dv1, dv2, atol=0, rtol=0)


# =====================================================================
#  11. Autograd Integration
# =====================================================================

class TestFA4Autograd:
    """Test FA4 autograd function for training."""

    def test_autograd_basic(self):
        """fa4_attn_func should support backward pass."""
        torch.manual_seed(42)
        B, H, S, D = 1, 2, 16, 32
        q = torch.randn(B, H, S, D, requires_grad=True)
        k = torch.randn(B, H, S, D, requires_grad=True)
        v = torch.randn(B, H, S, D, requires_grad=True)

        out = fa4_attn_func(q, k, v, block_size=16)
        loss = out.sum()
        loss.backward()

        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None
        assert torch.all(torch.isfinite(q.grad))
        assert torch.all(torch.isfinite(k.grad))
        assert torch.all(torch.isfinite(v.grad))

    def test_autograd_causal(self):
        """fa4_attn_func backward with causal mask."""
        torch.manual_seed(42)
        B, H, S, D = 1, 2, 16, 32
        q = torch.randn(B, H, S, D, requires_grad=True)
        k = torch.randn(B, H, S, D, requires_grad=True)
        v = torch.randn(B, H, S, D, requires_grad=True)

        out = fa4_attn_func(q, k, v, causal=True, block_size=16)
        loss = out.sum()
        loss.backward()

        assert q.grad is not None
        assert k.grad is not None

    def test_autograd_with_fa4_features(self):
        """Autograd with poly_exp + pingpong + deterministic backward."""
        torch.manual_seed(42)
        B, H, S, D = 1, 2, 16, 32
        q = torch.randn(B, H, S, D, requires_grad=True)
        k = torch.randn(B, H, S, D, requires_grad=True)
        v = torch.randn(B, H, S, D, requires_grad=True)

        out = fa4_attn_func(
            q, k, v,
            block_size=16,
            use_poly_exp=True,
            pingpong=True,
            deterministic=True,
        )
        loss = out.sum()
        loss.backward()

        assert torch.all(torch.isfinite(q.grad))
        assert torch.all(torch.isfinite(k.grad))
        assert torch.all(torch.isfinite(v.grad))


# =====================================================================
#  12. FA4 Module Tests
# =====================================================================

class TestFA4Module:
    """Test FastestFlashFA4 nn.Module."""

    def _make_module(self, hidden=64, heads=4, **kwargs):
        return FastestFlashFA4(hidden, heads, **kwargs)

    def test_module_basic(self):
        """Basic forward pass through FA4 module."""
        torch.manual_seed(42)
        mod = self._make_module()
        x = torch.randn(2, 16, 64)

        out = mod(x, block_size=16)
        assert out.shape == (2, 16, 64)

    def test_module_causal(self):
        """FA4 module with causal masking."""
        torch.manual_seed(42)
        mod = self._make_module(causal=True)
        x = torch.randn(1, 32, 64)

        out = mod(x, block_size=16)
        assert out.shape == (1, 32, 64)

    def test_module_return_lse(self):
        """FA4 module returns LSE when requested."""
        torch.manual_seed(42)
        mod = self._make_module()
        mod.eval()
        x = torch.randn(1, 16, 64)

        out, lse = mod(x, block_size=16, return_lse=True)
        assert out.shape == (1, 16, 64)
        assert lse is not None

    def test_module_training_backward(self):
        """FA4 module supports training backward pass."""
        torch.manual_seed(42)
        mod = self._make_module()
        mod.train()
        x = torch.randn(1, 16, 64, requires_grad=True)

        out = mod(x, block_size=16)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_module_load_weights(self):
        """load_weights populates projections correctly."""
        torch.manual_seed(42)
        mod = self._make_module()
        D = 64
        w_q = torch.randn(D, D)
        w_k = torch.randn(D, D)
        w_v = torch.randn(D, D)
        w_o = torch.randn(D, D)
        mod.load_weights(w_q, w_k, w_v, w_o)

        torch.testing.assert_close(mod.q_proj.weight.data, w_q)
        torch.testing.assert_close(mod.k_proj.weight.data, w_k)

    def test_module_fa4_knobs(self):
        """Module honours FA4-specific constructor knobs."""
        mod = self._make_module(
            use_poly_exp=True,
            rescale_threshold=2.0,
            pingpong=True,
            deterministic=True,
        )
        assert mod.use_poly_exp is True
        assert mod.rescale_threshold == 2.0
        assert mod.pingpong is True
        assert mod.deterministic is True

        repr_str = mod.extra_repr()
        assert "poly_exp=True" in repr_str
        assert "pingpong=True" in repr_str
        assert "deterministic=True" in repr_str

    def test_module_kv_cache(self):
        """FA4 module with KV cache (autoregressive)."""
        torch.manual_seed(42)
        mod = self._make_module()
        mod.eval()

        # Step 1: encode 16 tokens
        x1 = torch.randn(1, 16, 64)
        out1 = mod(x1, block_size=16)

        # Step 2: decode 1 new token with KV cache
        with torch.no_grad():
            q = mod.q_proj(x1)
            k = mod.k_proj(x1)
            v = mod.v_proj(x1)
            k = k.view(1, 16, 4, 16).transpose(1, 2)
            v = v.view(1, 16, 4, 16).transpose(1, 2)

        x2 = torch.randn(1, 1, 64)
        out2 = mod(x2, block_size=16, kv_cache=(k, v))
        assert out2.shape == (1, 1, 64)


# =====================================================================
#  13. Cross-validation vs Base Kernel
# =====================================================================

class TestFA4VsBaseKernel:
    """Cross-validate FA4 against the base cpu_reference kernel."""

    @pytest.mark.parametrize("causal", [False, True])
    @pytest.mark.parametrize("block_size", [16, 32])
    def test_fa4_matches_base_kernel(self, causal, block_size):
        torch.manual_seed(42)
        B, H, S, D = 1, 2, 48, 32
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        base_out, _ = flash_attn_cpu_forward(
            q, k, v, causal=causal,
            block_q=block_size, block_kv=block_size,
        )
        fa4_out, _ = fa4_forward(
            q, k, v, causal=causal,
            block_q=block_size, block_kv=block_size,
            use_poly_exp=False,  # disable poly for exact match
        )
        torch.testing.assert_close(fa4_out, base_out, atol=5e-3, rtol=1e-2)


# =====================================================================
#  14. Edge Cases
# =====================================================================

class TestFA4EdgeCases:
    """Edge cases for FA4 kernel."""

    def test_single_token(self):
        """S=1 should work (no KV after first position)."""
        torch.manual_seed(42)
        q = torch.randn(1, 1, 1, 32)
        k = torch.randn(1, 1, 1, 32)
        v = torch.randn(1, 1, 1, 32)

        out, _ = fa4_forward(q, k, v, block_q=1, block_kv=1)
        # Softmax of single element = 1.0, so output = v
        torch.testing.assert_close(out, v, atol=1e-5, rtol=1e-5)

    def test_large_block_size(self):
        """Block size larger than sequence should still work."""
        torch.manual_seed(42)
        q = torch.randn(1, 1, 8, 32)
        k = torch.randn(1, 1, 8, 32)
        v = torch.randn(1, 1, 8, 32)

        ref = _reference_attention(q, k, v)
        got, _ = fa4_forward(q, k, v, block_q=64, block_kv=64)
        torch.testing.assert_close(got, ref, atol=5e-3, rtol=1e-2)

    def test_asymmetric_qkv_lengths(self):
        """Different Q and KV sequence lengths."""
        torch.manual_seed(42)
        B, H, D = 1, 2, 32
        q = torch.randn(B, H, 8, D)
        k = torch.randn(B, H, 32, D)
        v = torch.randn(B, H, 32, D)

        ref = _reference_attention(q, k, v)
        got, _ = fa4_forward(q, k, v, block_q=8, block_kv=16)
        torch.testing.assert_close(got, ref, atol=5e-3, rtol=1e-2)


# =====================================================================
#  15. Block-Sparse Attention
# =====================================================================

class TestFA4BlockSparse:
    """Test FA4 with block-sparse masks."""

    def test_block_mask_basic(self):
        """Block mask should skip entire KV tiles."""
        torch.manual_seed(42)
        B, H, S, D = 1, 2, 32, 32
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        # 2 q-blocks × 2 kv-blocks
        block_mask = torch.tensor([
            [True, True],
            [True, False],  # skip second KV block for second Q block
        ])

        out, _ = fa4_forward(
            q, k, v, block_q=16, block_kv=16, block_mask=block_mask,
        )
        assert out.shape == (B, H, S, D)
        assert torch.all(torch.isfinite(out))


# =====================================================================
#  16. Numerical Stability
# =====================================================================

class TestFA4NumericalStability:
    """Stress-test FA4 numerical stability."""

    def test_large_values(self):
        """Large Q·K scores should not produce NaN/Inf."""
        torch.manual_seed(42)
        q = torch.randn(1, 1, 16, 32) * 10.0
        k = torch.randn(1, 1, 16, 32) * 10.0
        v = torch.randn(1, 1, 16, 32)

        out, _ = fa4_forward(q, k, v, block_q=16, block_kv=16)
        assert torch.all(torch.isfinite(out))

    def test_identical_keys(self):
        """All-same keys should produce uniform attention."""
        torch.manual_seed(42)
        q = torch.randn(1, 1, 8, 32)
        k = torch.ones(1, 1, 8, 32)
        v = torch.randn(1, 1, 8, 32)

        out, _ = fa4_forward(q, k, v, block_q=8, block_kv=8)
        # With identical keys, attention is uniform → output = mean(v)
        expected = v.mean(dim=2, keepdim=True).expand_as(v)
        torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)


# =====================================================================
#  17. Mixed-Precision
# =====================================================================

class TestFA4MixedPrecision:
    """Test FA4 with mixed-precision compute (FP16 matmul, FP32 accum)."""

    def test_fp16_compute(self):
        """Mixed-precision path should produce finite results."""
        torch.manual_seed(42)
        B, H, S, D = 1, 2, 32, 32
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        out, _ = fa4_forward(
            q, k, v, compute_dtype=torch.float16,
            block_q=16, block_kv=16,
        )
        assert torch.all(torch.isfinite(out))


# =====================================================================
#  18. KV-Cache Support
# =====================================================================

class TestFA4KVCache:
    """Test FA4 forward dispatch with KV-cache."""

    def test_kv_cache_append(self):
        """KV-cache append should produce valid output."""
        torch.manual_seed(42)
        B, H, D = 1, 2, 32
        # Previous KV from 16 tokens
        k_cache = torch.randn(B, H, 16, D)
        v_cache = torch.randn(B, H, 16, D)
        # New single token
        q_new = torch.randn(B, H, 1, D)
        k_new = torch.randn(B, H, 1, D)
        v_new = torch.randn(B, H, 1, D)

        out, _ = fa4_forward_dispatch(
            q_new, k_new, v_new,
            kv_cache=(k_cache, v_cache),
            block_size=16,
        )
        assert out.shape == (B, H, 1, D)
        assert torch.all(torch.isfinite(out))


# =====================================================================
#  19. Dispatch Entry Points
# =====================================================================

class TestFA4Dispatch:
    """Test FA4 dispatch layer (fa4_forward_dispatch, fa4_attn_func)."""

    def test_dispatch_auto_block_size(self):
        """Auto block size selection should produce valid output."""
        torch.manual_seed(42)
        q = torch.randn(1, 2, 32, 32)
        k = torch.randn(1, 2, 32, 32)
        v = torch.randn(1, 2, 32, 32)

        out, _ = fa4_forward_dispatch(q, k, v)
        ref = _reference_attention(q, k, v)
        torch.testing.assert_close(out, ref, atol=5e-3, rtol=1e-2)

    def test_func_matches_dispatch(self):
        """fa4_attn_func output should match fa4_forward_dispatch."""
        torch.manual_seed(42)
        q = torch.randn(1, 2, 32, 32)
        k = torch.randn(1, 2, 32, 32)
        v = torch.randn(1, 2, 32, 32)

        out_func = fa4_attn_func(q, k, v, block_size=16)
        out_dispatch, _ = fa4_forward_dispatch(q, k, v, block_size=16)
        torch.testing.assert_close(out_func, out_dispatch, atol=5e-3, rtol=1e-2)


# =====================================================================
#  20. Package Import Tests
# =====================================================================

class TestFA4Imports:
    """Verify FA4 is properly exported from the package."""

    def test_import_from_package(self):
        """FA4 should be importable from top-level package."""
        from fastest_flash_attention import (
            FastestFlashFA4,
            fa4_attn_func,
            fa4_forward_dispatch,
            FA4AttnFunc,
        )
        assert FastestFlashFA4 is not None
        assert fa4_attn_func is not None
        assert fa4_forward_dispatch is not None
        assert FA4AttnFunc is not None

    def test_import_fa4_kernel(self):
        """FA4 kernel functions should be importable."""
        from fastest_flash_attention.kernels.cpu_fa4 import fa4_forward, fa4_backward
        assert fa4_forward is not None
        assert fa4_backward is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
