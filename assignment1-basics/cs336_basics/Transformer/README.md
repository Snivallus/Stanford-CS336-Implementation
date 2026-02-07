# Transformer

This folder contains a minimal Transformer language model implementation built from scratch.  
It includes both the full language model (`TransformerLM`) and a collection of reusable 
Transformer building blocks.

---

## Folder Structure

```bash
Transformer/
├── README.md                     # This file
├── __init__.py                   # Module exports
├── transformer_lm.py             # Full Transformer language model (stack of blocks + LM head)
└── Basic_Building_Blocks/        # Core Transformer components
    ├── README.md
    ├── __init__.py
    ├── linear.py                         # Linear layer implementation
    ├── embedding.py                      # Token / positional embedding layers
    ├── rmsnorm.py                        # Root Mean Square Layer Normalization (RMSNorm) layer
    ├── swiglu.py                         # SwiGLU feed-forward block
    ├── rope.py                           # Rotary Positional Embeddings (RoPE)
    ├── softmax.py                        # Numerically stable softmax implementation
    ├── scaled_dot_product_attention.py   # Scaled dot-product attention primitive (with masking)
    ├── multihead_self_attention.py       # Causal multi-head self-attention module
    └── transformer_block.py              # Pre-norm Transformer block (Attention + FFN)
```