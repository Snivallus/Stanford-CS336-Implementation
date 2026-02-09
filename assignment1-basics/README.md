# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf).

---

## Setup

### Environment

Install dependencies:

```bash
uv sync
```

Run any Python file via:

```bash
uv run <python_file_path>
```

### Run unit tests

Run the standard test suite:

```bash
uv run pytest tests
```

Run the full test suite (including extended tests):

```bash
uv run pytest
```

---

## Folder Structure

```bash
assignment1-basics/
├── README.md                  # This file
├── Report.pdf                 # My report document
├── setup.py                   # CPython setup script
├── pyproject.toml             # Project + dependency configuration
├── cs336_basics/              # Main package source code
│   ├── README.md
│   ├── __init__.py
│   ├── BPE_Tokenizer/         # BPE tokenizer (training + encoding/decoding)
│   ├── Transformer/           # Transformer language model implementation
│   ├── Trainer/               # Training loop + checkpointing utilities
│   └── Generator/             # Autoregressive text generation / decoding
├── tests/                     # Unit tests
└── scripts/                   # Training / evaluation scripts
```