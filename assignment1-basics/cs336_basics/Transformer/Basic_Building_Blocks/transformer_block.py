import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from jaxtyping import Float, Int

from cs336_basics.Transformer.Basic_Building_Blocks import RMSNorm
from cs336_basics.Transformer.Basic_Building_Blocks import SwiGLU
from cs336_basics.Transformer.Basic_Building_Blocks import MultiHeadSelfAttention


class TransformerBlock(nn.Module):
    """
    Pre-Norm Transformer block with causal Multi-Head Self-Attention and SwiGLU FFN.

    This module implements the standard pre-norm Transformer block:

        y = x + MultiHeadSelfAttention(RMSNorm(x))
        z = y + SwiGLU(RMSNorm(y))

    Notes:
        - RMSNorm is applied before each sublayer (pre-norm).
        - Residual connections are applied after each sublayer.
        - MultiHeadSelfAttention is causal (prevents attending to future tokens).
        - SwiGLU is used as the feed-forward network.
    """


    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: Optional[int] = None,
        rope_theta: Optional[float] = None,
        max_seq_len: Optional[int] = None,
        eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize the Transformer block.

        Parameters:
            - d_model (int):
                Dimensionality of the Transformer block input/output.

            - num_heads (int):
                Number of attention heads.

            - d_ff (int | None):
                Hidden dimension used in the SwiGLU block.

            - rope_theta (float | None):
                RoPE Θ constant for rotary positional embeddings.
                If None, RoPE is disabled.

            - max_seq_len (int | None):
                Maximum sequence length supported by RoPE.
                Must be provided if rope_theta is provided.

            - eps (float):
                Epsilon value used in RMSNorm for numerical stability.

            - device (torch.device | None):
                Device on which parameters will be allocated.

            - dtype (torch.dtype | None):
                Datatype of the parameters.

        Attributes:
            - attn_norm (RMSNorm):
                RMSNorm layer applied before attention.
            - ffn_norm (RMSNorm):
                RMSNorm layer applied before feed-forward.
            - attn (MultiHeadSelfAttention):
                Causal multi-head self-attention sublayer.
            - ffn (SwiGLU):
                SwiGLU feed-forward sublayer.
        """
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        # initialize RMSNorm layers
        self.attn_norm = RMSNorm(
            d_model = d_model, 
            eps = eps, 
            device = device, 
            dtype = dtype
        )
        self.ffn_norm = RMSNorm(
            d_model = d_model, 
            eps = eps, 
            device = device, 
            dtype = dtype
        )
    
        # initialize MultiHeadSelfAttention layer
        self.attn = MultiHeadSelfAttention(
            d_model = d_model,
            num_heads = num_heads,
            rope_theta = rope_theta,
            max_seq_len = max_seq_len,
            device = device,
            dtype = dtype,
        )

        # initialize SwiGLU layer
        self.ffn = SwiGLU(
            d_model = d_model,
            d_ff = d_ff, 
            device = device, 
            dtype = dtype
        )

        # set d_ff attribute for statistics
        if d_ff is not None:
            self.d_ff = d_ff
        else:
            self.d_ff = self.ffn.d_ff


    def forward(
        self,
        x: Float[Tensor, "... seq_len d_model"],
        token_positions: Optional[Int[Tensor, "... seq_len"]] = None,
    ) -> Float[Tensor, "... seq_len d_model"]:
        """
        Apply the Transformer block.

        Parameters:
            - x (Tensor):
                Input tensor of shape (..., seq_len, d_model).

            - token_positions (Tensor | None):
                Token positions of shape (..., seq_len).
                Only required if RoPE is enabled.

        Returns:
            - output (Tensor):
                Output tensor of shape (..., seq_len, d_model).
        """
        # First sublayer: RMSNorm -> MHA -> residual
        y = x + self.attn(self.attn_norm(x), token_positions=token_positions)

        # Second sublayer: RMSNorm -> FFN -> residual
        z = y + self.ffn(self.ffn_norm(y))

        return z