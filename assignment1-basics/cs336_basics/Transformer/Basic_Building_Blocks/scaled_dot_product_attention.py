import math
import torch
from torch import Tensor
from typing import Optional
from jaxtyping import Float, Bool
from einops import einsum

from cs336_basics.Transformer.Basic_Building_Blocks import softmax


def scaled_dot_product_attention(
    Q: Float[Tensor, "... seq_len d_k"],
    K: Float[Tensor, "... seq_len d_k"],
    V: Float[Tensor, "... seq_len d_v"],
    mask: Optional[Bool[Tensor, "... seq_len seq_len"]] = None,
) -> Float[Tensor, "... seq_len d_v"]:
    """
    Compute scaled dot-product attention:

        Attention(Q, K, V) = softmax((Q K^T) / sqrt(d_k)) V

    This implementation supports arbitrary batch dimensions and an optional
    boolean attention mask.

    Parameters:
        - Q (Tensor):
            Query tensor of shape (..., seq_len, d_k).

        - K (Tensor):
            Key tensor of shape (..., seq_len, d_k).

        - V (Tensor):
            Value tensor of shape (..., seq_len, d_v).

        - mask (Tensor | None):
            Optional boolean tensor of shape (seq_len, seq_len).
            mask[i, j] = True means query position i can attend to key position j.
            mask[i, j] = False means attention is blocked at that position.

    Returns:
        - output (Tensor):
            Attention output tensor of shape (..., seq_len, d_v).

    Notes:
        - Uses the numerically stable softmax trick by subtracting the max value.
        - Masked positions are assigned -inf before softmax, ensuring their
          attention probability becomes 0.
        - Uses einops.einsum for explicit tensor contractions.
    """
    d_k = Q.shape[-1]
    scale = 1.0 / math.sqrt(d_k)

    # compute attention logits: (..., seq_len, seq_len)
    attn_logits = einsum(Q, K, "... q d_k, ... k d_k -> ... q k") * scale

    if mask is not None:
        # ensure mask is broadcastable to (..., seq_len, seq_len)
        attn_logits = attn_logits.masked_fill(~mask, float("-inf"))

    # softmax along key dimension
    attn_weights = softmax(
        x = attn_logits,
        dim = -1,
    )

    # weighted sum over values: (..., seq_len, d_v)
    out = einsum(attn_weights, V, "... q k, ... k d_v -> ... q d_v")
    return out