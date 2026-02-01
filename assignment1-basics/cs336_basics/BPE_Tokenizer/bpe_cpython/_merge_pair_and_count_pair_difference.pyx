# cython: boundscheck = False, wraparound = False
"""
Native (Cython) implementation of the core hot loop used during BPE training.

This file provides a CPython/Cython-accelerated implementation of
`BPE_Trainer._merge_pair_and_count_pair_difference` from `train_bpe.py`.

The function defined here is invoked during each BPE merge step for every
"affected word" and is therefore performance-critical.

Design goals:
- Move the most expensive inner loops (token scanning and merging) out of Python.
- Avoid Python-level Counter and dict operations in the hot path.
- Return results in a flat, Python-friendly structure that can be efficiently
  consumed by higher-level Python code.
- Keep the interface compatible with a pure-Python fallback implementation.
"""

from libc.stdlib cimport malloc, free


def _merge_pair_and_count_pair_difference(
    list old_encoding,
    int bytes_a,
    int bytes_b,
    int new_id,
):
    """
    Merge a specific token pair inside a single word encoding and collect
    pair-count differences before and after the merge.

    This function operates on a *single word* and performs three tasks:
    1. Count all adjacent token pairs in the original encoding.
    2. Merge occurrences of the target pair (bytes_a, bytes_b) into new_id.
    3. Count all adjacent token pairs in the new encoding.

    Parameters
    ----------
    old_encoding : list[int]
        Token IDs representing the original encoding of a word.
    bytes_a : int
        First token ID of the pair to be merged.
    bytes_b : int
        Second token ID of the pair to be merged.
    new_id : int
        Token ID assigned to the merged pair.

    Returns
    -------
    new_encoding : list[int]
        The word encoding after merging (bytes_a, bytes_b) into new_id.
    old_pairs : list[tuple[int, int, int]]
        Flat list of (token_a, token_b, count) for pairs in the original encoding.
        Each entry represents a single occurrence (count = 1).
    new_pairs : list[tuple[int, int, int]]
        Flat list of (token_a, token_b, count) for pairs in the new encoding.
    """

    cdef Py_ssize_t n = len(old_encoding)
    cdef Py_ssize_t i

    old_pairs = []
    new_pairs = []

    # ---------------------------------------------------------------------
    # Step 1: Count adjacent token pairs in the original encoding.
    #
    # We record each occurrence explicitly instead of aggregating counts here.
    # This avoids hash table operations in the hot path and allows the Python
    # layer to combine counts with word frequency later.
    # ---------------------------------------------------------------------
    for i in range(n - 1):
        old_pairs.append((old_encoding[i], old_encoding[i + 1], 1))

    # ---------------------------------------------------------------------
    # Step 2: Merge the target pair (bytes_a, bytes_b) into new_id.
    #
    # This is a single left-to-right scan of the token list.
    # When the target pair is found, it is replaced by new_id and the scan
    # advances by two positions; otherwise, the current token is copied.
    # ---------------------------------------------------------------------
    new_encoding = []
    i = 0
    while i < n:
        if (
            i < n - 1
            and old_encoding[i] == bytes_a
            and old_encoding[i + 1] == bytes_b
        ):
            # Merge the pair into a single new token.
            new_encoding.append(new_id)
            i += 2
        else:
            # Copy the original token unchanged.
            new_encoding.append(old_encoding[i])
            i += 1

    # ---------------------------------------------------------------------
    # Step 3: Count adjacent token pairs in the new encoding.
    #
    # As with old_pairs, we emit flat (token_a, token_b, 1) tuples and defer
    # aggregation to the Python layer.
    # ---------------------------------------------------------------------
    cdef Py_ssize_t n2 = len(new_encoding)
    for i in range(n2 - 1):
        new_pairs.append((new_encoding[i], new_encoding[i + 1], 1))

    return old_pairs, new_encoding, new_pairs