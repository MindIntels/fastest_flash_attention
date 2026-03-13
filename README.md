# fastest Flash Attention

Production-ready flash attention implementation combining algorithmic innovations
from [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) (FlashAttention-2/3/4)
and [xlite-dev/ffpa-attn](https://github.com/xlite-dev/ffpa-attn) (Fine-Grained Pipelined Flash Attention).

## Architecture

```
fastest_flash_attention/
├── __init__.py                     # Public API
├── config.py                       # AttentionConfig dataclass
├── utils.py                        # Helpers (auto block-size, repeat_kv, etc.)
├── kernels/
│   ├── cpu_reference.py            # Pure-PyTorch CPU reference (fwd + bwd)
│   ├── cpu_fa4.py                  # FlashAttention-4 CPU reference (fwd + bwd)
│   ├── triton_fwd.py               # Triton GPU forward kernel
│   ├── triton_bwd.py               # Triton GPU backward kernel
│   └── triton_ffpa.py              # FFPA-style Triton kernel (auto-tuned)
├── ops/
│   ├── attention_forward.py        # Forward dispatch (auto CPU/GPU)
│   ├── attention_backward.py       # Backward dispatch
│   ├── attention_func.py           # torch.autograd.Function wrapper
│   ├── fa4_func.py                 # FA4 autograd.Function + functional APIs
│   ├── kv_cache.py                 # Continuous + Paged KV-cache
│   └── block_sparse.py             # Block-mask generators
├── modules/
│   ├── flash_mha.py                # Multi-Head Attention nn.Module
│   ├── flash_gqa.py                # Grouped-Query Attention nn.Module
│   ├── flash_mla.py                # Multi-Latent Attention nn.Module (DeepSeek)
│   └── flash_fa4.py                # FlashAttention-4 nn.Module
├── tests/
│   ├── test_fastest_flash.py       # Core flash attention tests (50+ cases)
│   └── test_fa4.py                 # FlashAttention-4 tests (64 cases)
├── run_tests.sh                    # Run all tests
└── README.md
```

## Key Features

| Feature | Source | Description |
|---------|--------|-------------|
| Tiled Online Softmax | FlashAttention-1 | O(block) memory, no full S×S matrix |
| Deferred Rescaling | FlashAttention-2 | Single normalisation per Q-block |
| Mixed-Precision | FlashAttention-3 | FP16/BF16 compute, FP32 accumulation |
| Two-Pass Softmax | FlashAttention-3 | Exact logsumexp for numerical precision |
| Ping-Pong Pipeline | FlashAttention-3/4 | Simulated warp-specialised scheduling |
| **Software Exp2 Polynomial** | **FlashAttention-4** | Horner-form polynomial emulating 2^x on tensor cores |
| **Conditional Rescaling** | **FlashAttention-4** | Skip costly rescale when max-shift < threshold τ |
| **Ping-Pong Q Tiles** | **FlashAttention-4** | Alternate Q tiles across warps to hide latency |
| **LPT Scheduling** | **FlashAttention-4** | Longest-processing-time-first tile ordering for causal |
| **2-CTA Backward** | **FlashAttention-4** | Dual-CTA simulation for deterministic backward |
| **Deterministic Backward** | **FlashAttention-4** | Bit-exact gradient reproducibility |
| **Fine-Grained Pipeline** | **FFPA** | Double-buffered KV loading overlapped with compute |
| **Flat GEMM Decomposition** | **FFPA** | Sub-tile Q·K^T for better instruction-level parallelism |
| **Auto-Tuning** | **FFPA** | Triton autotuner selects best block config |
| Causal Masking | Both | Block-level early-exit saves ~50% tiles |
| Sliding Window | FlashAttention | Local attention with configurable window |
| Softcap | Gemma-2 | `softcap * tanh(score / softcap)` logit capping |
| Block-Sparse | FlashAttention-3 | Skip tiles via configurable block masks |
| Paged KV-Cache | vLLM | Memory-efficient autoregressive decoding |
| GQA / MQA | LLaMA-2+ | Grouped-query and multi-query attention |
| MLA | DeepSeek-V2 | Low-rank KV compression + decoupled RoPE |
| Autograd Support | Custom | Memory-efficient backward with recomputation |

## Quick Start

### Functional API

```python
import torch
from fastest_flash_attention import fastest_flash_attn_forward, fastest_flash_attn_func

# Basic usage
B, H, S, D = 2, 8, 1024, 64
q = torch.randn(B, H, S, D)
k = torch.randn(B, H, S, D)
v = torch.randn(B, H, S, D)

# Forward only (inference)
output, lse = fastest_flash_attn_forward(q, k, v, causal=True)

# Differentiable (training)
output = fastest_flash_attn_func(q, k, v, causal=True, block_size=64)
loss = output.sum()
loss.backward()  # Memory-efficient backward via recomputation
```

### FA4 Functional API

```python
from fastest_flash_attention import fa4_attn_func, fa4_forward_dispatch

# Forward only with FA4 features
output, lse = fa4_forward_dispatch(
    q, k, v,
    causal=True,
    use_poly_exp=True,       # polynomial exp2 emulation
    rescale_threshold=0.5,   # conditional rescaling threshold τ
    pingpong=True,           # ping-pong Q tile scheduling
)

# Differentiable (training)
output = fa4_attn_func(
    q, k, v,
    causal=True,
    block_size=64,
    use_poly_exp=True,
    deterministic=True,      # bit-exact backward
)
loss = output.sum()
loss.backward()
```

### Module API

```python
from fastest_flash_attention import FastestFlashMHA, FastestFlashGQA, FastestFlashMLA, FastestFlashFA4

# Standard MHA
mha = FastestFlashMHA(hidden_size=512, num_heads=8, causal=True)
out = mha(x)  # x: [B, S, 512]

# GQA (LLaMA-style: 32 Q heads, 8 KV heads)
gqa = FastestFlashGQA(hidden_size=4096, num_heads=32, num_kv_heads=8, causal=True)
out = gqa(x)

# MLA (DeepSeek-style: low-rank KV compression)
mla = FastestFlashMLA(hidden_size=4096, num_heads=32, latent_dim=512, rope_dim=64)
out = mla(x)

# FA4 (FlashAttention-4: poly exp2, conditional rescaling, ping-pong)
fa4 = FastestFlashFA4(
    hidden_size=4096, num_heads=32,
    causal=True,
    use_poly_exp=True,
    rescale_threshold=0.0,   # 0.0 = always exact (default)
    pingpong=True,
    deterministic=False,
)
out = fa4(x, block_size=64)
```

### With KV-Cache (Autoregressive Decoding)

```python
from fastest_flash_attention import ContinuousKVCache, PagedKVCache

# Continuous cache
cache = ContinuousKVCache(max_batch=4, max_seq_len=2048, num_heads=8, head_dim=64)
k_full, v_full = cache.append(k_new, v_new)

# Paged cache (vLLM-style)
paged_cache = PagedKVCache(max_pages=1024, page_size=64, num_heads=8, head_dim=64)
paged_cache.append_tokens(seq_id=0, k_tokens=k, v_tokens=v)
k_out, v_out = paged_cache.get_kv(seq_id=0)
```

### With Config

```python
from fastest_flash_attention import AttentionConfig, fastest_flash_attn_forward

config = AttentionConfig(
    causal=True,
    sliding_window=256,
    softcap=30.0,
    block_size_q=128,
    block_size_kv=64,
    use_triton=True,
    pipeline_stages=2,
)

output, lse = fastest_flash_attn_forward(q, k, v, config=config)
```

## Auto-Dispatch

The library automatically selects the best kernel:

1. **CUDA + Triton available** → FFPA auto-tuned Triton kernel
2. **CUDA + Triton, complex features** → Standard Triton kernel
3. **CPU / no Triton** → Optimised CPU reference kernel

All kernels produce numerically equivalent results (within floating-point tolerance).

## Running Tests

```bash
# Run all tests
cd fastest_flash_attention
bash run_tests.sh

# Or run individually
python -m pytest tests/test_fastest_flash.py -v    # Core attention (50+ tests)
python -m pytest tests/test_fa4.py -v              # FlashAttention-4 (64 tests)

# Run all tests with verbose output
python -m pytest tests/ -v --tb=short
```

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0.0
- pytest ≥ 7.0 (tests)
- triton ≥ 2.1.0 (optional, for GPU kernels)

## Algorithm Details

### Forward Pass (FFPA-Enhanced)

```
for each Q-block i:
    O_i = 0, lse_i = -inf                    # FP32 accumulators
    prefetch K[0], V[0]                       # FFPA: async load
    for each KV-block j:
        [skip if block_mask / causal / window]
        K_j, V_j = prefetch buffer            # FFPA: use prefetched data
        async_load K[j+1], V[j+1]             # FFPA: overlap with compute
        S_ij = Q_i · K_j^T × scale            # mixed-precision matmul
        S_ij = softcap * tanh(S_ij/softcap)   # optional softcap
        [apply causal/window mask]
        m_ij, P_ij, lse_ij = online_softmax(S_ij)
        O_i = rescale(O_i, lse_i, lse_ij) + P_ij · V_j
        lse_i = logaddexp(lse_i, lse_ij)      # deferred rescaling
    output[i] = O_i                           # single division at end
```

### FA4 Forward Pass (FlashAttention-4 Enhancements)

```
tile_order = LPT_schedule(n_br, causal)       # reverse for causal balance
for i in tile_order:                           # ping-pong: alternate Q tiles
    O_i = 0, lse_i = -inf, m_prev = -inf
    for each KV-block j:
        [skip if block_mask / causal / window]
        S_ij = Q_i · K_j^T × scale
        [apply causal/window/softcap mask]
        m_ij = rowmax(S_ij)
        max_jump = |m_ij - m_prev|
        if max_jump < τ:                       # conditional rescaling
            P_ij = software_exp2((S_ij - m_cur) / ln2)  # poly Horner exp2
            O_i += P_ij · V_j                  # skip rescale (≈ free)
        else:
            rescale_factor = exp(m_prev - m_ij)
            O_i = O_i * rescale_factor + P_ij · V_j
        m_prev = m_ij
    output[i] = O_i / rowsum                   # final normalisation
```

### Backward Pass (Memory-Efficient)

```
for each KV-block j:
    dK_j = 0, dV_j = 0
    for each Q-block i:
        Recompute: S_ij = Q_i · K_j^T × scale
        P_ij = exp(S_ij - stored_lse)         # use stored logsumexp
        dV_j += P_ij^T · dO_i
        dP_ij = dO_i · V_j^T
        D_i = rowsum(dO_i * O_i)
        dS_ij = P_ij * (dP_ij - D_i)
        dQ_i += dS_ij · K_j × scale
        dK_j += dS_ij^T · Q_i × scale
```

### FA4 Backward (2-CTA Deterministic)

```
# CTA-0: compute dV (accumulate in shared memory)
# CTA-1: compute dQ, dK (accumulate per-block, reduce)
for each KV-block j:
    for each Q-block i:
        Recompute S_ij, P_ij using stored lse
        # CTA-0
        dV_j += P_ij^T · dO_i
        # CTA-1
        dP_ij = dO_i · V_j^T
        D_i = rowsum(dO_i * O_i)
        dS_ij = P_ij * (dP_ij - D_i)
        dQ_i += dS_ij · K_j × scale   # deterministic: fixed reduce order
        dK_j += dS_ij^T · Q_i × scale
```

## License

MIT
