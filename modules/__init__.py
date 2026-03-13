"""
Fastest Flash Attention — Module wrappers.
"""

from .flash_mha import FastestFlashMHA
from .flash_gqa import FastestFlashGQA
from .flash_mla import FastestFlashMLA
from .flash_fa4 import FastestFlashFA4

__all__ = ["FastestFlashMHA", "FastestFlashGQA", "FastestFlashMLA", "FastestFlashFA4"]
