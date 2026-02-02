"""
CPython-accelerated BPE utilities.

This subpackage provides a Cython-based implementation of
_merge_pair_and_count_pair_difference for faster BPE training.
If the compiled extension is unavailable, it falls back to
the pure Python implementation.
"""

from ._merge_pair_and_count_pair_difference import _merge_pair_and_count_pair_difference