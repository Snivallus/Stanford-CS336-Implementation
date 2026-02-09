# CS336: Language Modeling from Scratch

This repository contains my implementation of the programming assignments for
[Stanford CS336 (Spring 2025)](https://stanford-cs336.github.io/spring2025/).

---

## Prerequisites

- **Python:** 3.11+
- **Environment manager:** [uv](https://docs.astral.sh/uv/guides/projects/#managing-dependencies)

Install `uv` and download datasets:

```bash
pip install uv
source download_datasets.sh
```

Each assignment is self-contained.   
Enter the assignment folder and install dependencies, for example:

```bash
cd assignment1-basics
uv sync
```

---

## Project Structure

```bash
Stanford-CS336-Implementation/
├── README.md                      # This file
├── download_datasets.sh           # Dataset download script
├── datasets/                      # Shared datasets
│   ├── TinyStoriesV2-GPT4-train.txt
│   ├── TinyStoriesV2-GPT4-valid.txt
│   ├── owt_train.txt
│   └── owt_valid.txt
├── assignment1-basics/            # Transformer LM from scratch
│   ├── README.md
│   ├── Report.pdf
│   ├── setup.py
│   ├── pyproject.toml
│   ├── cs336_basics/              # Main package source code
│   ├── tests/                     # Unit tests
│   └── scripts/                   # Training / evaluation scripts
├── assignment2-systems/           # TODO
├── assignment3-scaling/           # TODO
├── assignment4-data/              # TODO
└── assignment5-alignment/         # TODO
```