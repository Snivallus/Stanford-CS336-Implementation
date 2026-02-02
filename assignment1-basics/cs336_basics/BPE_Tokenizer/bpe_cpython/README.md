# BPE Tokenizer — CPython Extension

This folder contains a **CPython-accelerated implementation** of the core pair-merging function used in the Byte Pair Encoding (BPE) training pipeline. It is part of the `cs336_basics` package and is intended to significantly speed up the `_merge_pair_and_count_pair_difference` computation during BPE vocabulary learning.

---

## Folder Structure

After running `uv sync` in the root folder of CS336 Assignment 1, the folder should look like this:

```bash
bpe_cpython/
├── README.md # This file
├── init.py
├── _merge_pair_and_count_pair_difference.pyx # Cython source
├── _merge_pair_and_count_pair_difference.c # Generated C source
└── _merge_pair_and_count_pair_difference.cpython-311-x86_64-linux-gnu.so # Compiled extension
```

- **`_merge_pair_and_count_pair_difference.pyx`**  
  The Cython source implementing `_merge_pair_and_count_pair_difference`. It contains the CPython-level optimized version of the algorithm.

- **`_merge_pair_and_count_pair_difference.c`**  
  Auto-generated C source from the `.pyx` file. Generated when compiling with `Cython`.

- **`_merge_pair_and_count_pair_difference.cpython-311-x86_64-linux-gnu.so`**  
  The compiled native extension. Loaded at runtime for accelerated performance.

---

## Installation / Compilation

To build the Cython extension locally:

```bash
cd cs336_basics/BPE_Tokenizer/bpe_cpython
uv run python setup.py build_ext --inplace
```

- This will generate or update the `.so` file in-place. 

- If the build fails or the `.so` file is missing, the Python fallback will automatically be used.