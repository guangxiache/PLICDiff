"""Compatibility shim preserving the legacy import path: from extract_PLIC import PLIC.

The implementation lives in datasets/extract_PLIC.py and is shared by training and sampling.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from datasets.extract_PLIC import PLIC  # noqa: F401
