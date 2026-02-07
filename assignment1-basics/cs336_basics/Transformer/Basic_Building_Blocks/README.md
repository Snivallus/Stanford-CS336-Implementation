# Basic_Building_Blocks

This folder contains core Transformer building blocks used throughout the assignment.

---

## Folder Structure

```bash
Basic_Building_Blocks/
├── README.md                         # This file
├── __init__.py                       # Module exports
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

Files are arranged roughly in the order they are implemented and composed.  
Later modules depend on earlier primitives (e.g., MHA depends on attention + softmax + RoPE).