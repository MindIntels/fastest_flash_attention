"""
Fastest Flash Attention — Kernel implementations.

Provides:
  - cpu_reference: Pure-PyTorch CPU reference kernel.
  - triton_fwd:    Triton GPU forward kernel.
  - triton_bwd:    Triton GPU backward kernel.
  - triton_ffpa:   FFPA-style fine-grained pipelined Triton kernel.
"""

from .cpu_reference import (
    flash_attn_cpu_forward,
    flash_attn_cpu_backward,
)
from .cpu_fa4 import (
    fa4_forward,
    fa4_backward,
)

__all__ = [
    "flash_attn_cpu_forward",
    "flash_attn_cpu_backward",
    "fa4_forward",
    "fa4_backward",
]

# Conditionally export Triton kernels
try:
    import triton  # noqa: F401
    from .triton_fwd import flash_attn_triton_forward
    from .triton_bwd import flash_attn_triton_backward
    from .triton_ffpa import ffpa_attn_triton_forward
    __all__ += [
        "flash_attn_triton_forward",
        "flash_attn_triton_backward",
        "ffpa_attn_triton_forward",
    ]
except ImportError:
    pass
