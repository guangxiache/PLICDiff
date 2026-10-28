"""Legacy import path for the shared PLIC extractor."""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from datasets.extract_PLIC import PLIC

__all__ = ['PLIC']
