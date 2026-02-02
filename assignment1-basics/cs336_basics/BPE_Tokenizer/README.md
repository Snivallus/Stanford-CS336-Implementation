# BPE_Tokenizer

This folder contains an implementation of **Byte Pair Encoding (BPE) training** for tokenization, developed as part of **Stanford CS336 (Spring 2025) – Assignment 1 (Basics)**.

---

## Folder Structure

```bash
BPE_Tokenizer/
├── README.md          # This file
├── __init__.py
├── train_bpe.py       # Main BPE training logic
├── max_heapq.py       # Compatibility max-heap wrapper around heapq
├── bpe_cpython/       # Optional CPython acceleration
│   └── README.md
└── tests/             # Tests for the BPE implementation
    └── train_bpe_test.py
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