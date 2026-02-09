# cs336_basics

This folder contains the main implementation code for **CS336 Spring 2025 Assignment 1**.  
It includes a complete pipeline for training and sampling from a Transformer language model,
starting from tokenizer training to model training and autoregressive generation.

---

## Folder Structure

```bash
cs336_basics/
├── README.md          # This file
├── __init__.py        # Package exports
├── BPE_Tokenizer/     # Byte Pair Encoding (BPE) tokenizer implementation
│   └── README.md
├── Transformer/       # Transformer language model implementation (decoder-only)
│   └── README.md
├── Trainer/           # Training loop + optimizer/scheduler/loss/checkpoint utilities
│   └── README.md
└── Generator/         # Autoregressive decoding / text generation utilities
    └── README.md
```