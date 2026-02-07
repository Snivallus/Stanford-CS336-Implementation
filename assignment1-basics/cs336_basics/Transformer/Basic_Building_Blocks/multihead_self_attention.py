import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from jaxtyping import Float, Int
from einops import rearrange

from cs336_basics.Transformer.Basic_Building_Blocks import Linear
from cs336_basics.Transformer.Basic_Building_Blocks import RoPE
from cs336_basics.Transformer.Basic_Building_Blocks import scaled_dot_product_attention


class MultiHeadSelfAttention(nn.Module):
    """
    Causal Multi-Head Self-Attention module with RoPE.

    This module implements:

        MultiHeadSelfAttention(x) = W_O * MultiHead(W_Q x, W_K x, W_V x)

    where the multi-head attention is computed by splitting the projected Q, K, V
    tensors into `num_heads` independent heads.

    Notes:
        - Uses causal masking to prevent attending to future tokens.
        - Uses RoPE (Rotary Positional Embeddings) on query and key vectors only.
        - Uses custom Linear modules (not nn.Linear).
        - Assumes d_k = d_v = d_model / num_heads.
    """


    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope_theta: Optional[float] = None,
        max_seq_len: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize the causal multi-head self-attention module.

        Parameters:
            - d_model (int):
                Dimensionality of the Transformer block input/output.

            - num_heads (int):
                Number of attention heads.

            - rope_theta (float | None):
                RoPE Θ constant.

            - max_seq_len (int | None):
                Maximum sequence length supported for RoPE.

            - device (torch.device | None):
                Device on which the parameters/buffers will be allocated.

            - dtype (torch.dtype | None):
                Datatype of the parameters.

        Attributes:
            - d_k (int):
                Per-head key/query dimension.
            - d_v (int):
                Per-head value dimension.
            - w_qkv (Linear):
                Combined projection for Q, K, V.
            - w_o (Linear):
                Output projection.
            - rope (RotaryPositionalEmbedding):
                Shared RoPE module for Q and K.
        """
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model must be divisible by num_heads, got d_model={d_model}, num_heads={num_heads}"
            )

        self.d_model = d_model
        self.h = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        # Single matrix multiply to produce Q, K, V
        self.w_qkv = Linear(
            in_features = d_model, 
            out_features = num_heads * (2 * self.d_k + self.d_v), 
            device = device, 
            dtype = dtype
        )

        # Output projection
        self.w_o = Linear(
            in_features = num_heads * self.d_v, 
            out_features = d_model, 
            device = device, 
            dtype = dtype
        )

        # Optional RoPE, which is applied to Q and K only
        self.rope = None
        if rope_theta is not None and max_seq_len is not None:
            self.rope = RoPE(
                rope_theta = rope_theta,
                d_k = self.d_k,
                max_seq_len = max_seq_len,
                device = device,
            )


    def forward(
        self, 
        x: Float[Tensor, "... seq_len d_model"],
        token_positions: Optional[Int[Tensor, "... seq_len"]] = None,
    ) -> Float[Tensor, "... seq_len d_model"]:
        """
        Apply causal multi-head self-attention to the input.

        Parameters:
            - x (Tensor):
                Input tensor of shape (..., seq_len, d_model).

            - token_positions (Tensor | None):
                Token positions of shape (..., seq_len).

        Returns:
            - output (Tensor):
                Tensor of shape (..., seq_len, d_model).

        Notes:
            - Applies causal masking so each token attends only to itself and past tokens.
            - Applies RoPE to query and key vectors for each head.
        """
        seq_len = x.shape[-2]

        # Project into Q, K, V: (..., seq_len, h * (2 * d_k + d_v))
        qkv = self.w_qkv(x)

        # Split: (..., seq_len, h * d_k), (..., seq_len, h * d_k), (..., seq_len, h * d_v)
        Q, K, V = torch.split(qkv, [self.h * self.d_k, self.h * self.d_k, self.h * self.d_v], dim=-1)

        # Reshape into heads: 
        # (..., h, seq_len, d_k), (..., h, seq_len, d_k), (..., h, seq_len, d_v)
        Q = rearrange(Q, "... s (h d_k) -> ... h s d_k", h=self.h)
        K = rearrange(K, "... s (h d_k) -> ... h s d_k", h=self.h)
        V = rearrange(V, "... s (h d_v) -> ... h s d_v", h=self.h)

        # Apply RoPE to Q and K (broadcast token_positions across heads)
        if self.rope is not None:
            # create token_positions tensor if not provided
            if token_positions is None:
                token_positions = torch.arange(
                    seq_len, device=x.device, dtype=torch.int64
                ).expand(x.shape[:-1])
            Q = self.rope(Q, token_positions.unsqueeze(-2))
            K = self.rope(K, token_positions.unsqueeze(-2))

        # Causal mask: (seq_len, seq_len)
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
        )

        # Attention per head: (..., h, seq_len, d_v)
        attn_out = scaled_dot_product_attention(Q, K, V, mask=causal_mask)

        # Merge heads: (..., seq_len, h*d_v)
        attn_out = rearrange(attn_out, "... h s d -> ... s (h d)")

        # Output projection: (..., seq_len, d_model)
        return self.w_o(attn_out)