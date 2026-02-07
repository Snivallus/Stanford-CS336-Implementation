# BPE_Tokenizer

This folder contains an implementation of Byte Pair Encoding (BPE) trainer and tokenizer.

---

## Folder Structure

```bash
BPE_Tokenizer/
├── README.md          # This file
├── __init__.py        # Module exports
├── train_bpe.py       # BPE trainer
├── tokenizer.py       # BPE tokenizer
├── max_heapq.py       # Compatibility max-heap wrapper around heapq
├── bpe_cpython/       # Optional CPython acceleration
│   ├── README.md
│   └── _merge_pair_and_count_pair_difference.pyx
└── tests/             # Tests for the BPE implementation
    ├── README.md
    ├── test_01_train_bpe.py
    └── test_02_tokenizer.py
```

---

## Running Tests

Run the standard test suite from the project root:

```bash
uv run pytest tests/test_train_bpe.py
```

Run extended tests (results are saved to `cs336_basics/BPE_Tokenizer/tests`):

```bash
uv run pytest cs336_basics/BPE_Tokenizer/tests
```