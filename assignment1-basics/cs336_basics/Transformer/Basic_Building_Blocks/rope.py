import torch
import torch.nn as nn
from typing import Optional
from jaxtyping import Float, Int
# from einops import rearrange


class RoPE(nn.Module):
    """
    Rotary Positional Embedding (RoPE) module.

    RoPE injects positional information by applying a position-dependent rotation
    to the last dimension of the input tensor. For each position i and each pair
    of features (2k, 2k+1), the rotation angle is:

        theta_{i,k} = i / Theta^(2k / d_k)

    and the transformation is:

        [x_even']   [ cos(theta)  -sin(theta) ] [x_even]
        [x_odd' ] = [ sin(theta)   cos(theta) ] [x_odd ]

    Notes:
        - RoPE is applied over the last dimension (d_k), which must be even.
        - Cosine and sine tables are precomputed up to max_seq_len and stored
          as non-persistent buffers (not learnable).
        - Supports arbitrary batch dimensions.
    """


    def __init__(
        self,
        rope_theta: float,
        d_k: int,
        max_seq_len: int,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the RoPE module.

        Parameters:
            - rope_theta (float):
                The Θ constant used in the frequency computation.

            - d_k (int):
                Dimension of the query/key vectors. Must be even.

            - max_seq_len (int):
                Maximum sequence length that will be inputted.

            - device (torch.device | None):
                Device on which the precomputed buffers will be stored.

        Attributes:
            - cos (torch.Tensor):
                Precomputed cosine table of shape (max_seq_len, d_k/2).
            - sin (torch.Tensor):
                Precomputed sine table of shape (max_seq_len, d_k/2).
        """
        super().__init__()

        if d_k % 2 != 0:
            raise ValueError(f"d_k must be even for RoPE, but got d_k = {d_k}")

        self.rope_theta = rope_theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # Compute inverse frequencies:
        # inv_freq[k] = 1 / Theta^(2k/d_k)
        half_dim = d_k // 2
        k = torch.arange(half_dim, device=device, dtype=torch.float32)
        inv_freq = rope_theta ** (-2.0 * k / d_k)

        # Positions: (max_seq_len,)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)

        # angles[i, k] = i * inv_freq[k]
        angles = positions[:, None] * inv_freq[None, :]  # (max_seq_len, half_dim)

        cos = torch.cos(angles)
        sin = torch.sin(angles)

        # Store as buffers (not learnable, not persistent in state_dict)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)


    def forward(
        self, 
        x: Float[torch.Tensor, "... seq_len d_k"],
        token_positions: Int[torch.Tensor, "... seq_len"],
    ) -> Float[torch.Tensor, "... seq_len d_k"]:
        """
        Apply RoPE rotation to the input tensor.

        Parameters:
            - x (torch.Tensor):
                Input tensor of shape (..., seq_len, d_k).

            - token_positions (torch.Tensor):
                Tensor of shape (..., seq_len) specifying the position index
                of each token along the sequence dimension.

        Returns:
            - output (torch.Tensor):
                Tensor of shape (..., seq_len, d_k), with RoPE applied.

        Notes:
            - Uses token_positions to index into the precomputed cos/sin tables.
            - Supports arbitrary batch dimensions.
        """
        # seq_len = x.shape[-2]
        d_k = x.shape[-1]

        if d_k != self.d_k:
            raise ValueError(f"Expected last dim d_k={self.d_k}, but got {d_k}")

        if token_positions.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"token_positions must be int32 or int64, got {token_positions.dtype}")

        # Check token_positions bounds
        max_pos = int(token_positions.max().item())
        min_pos = int(token_positions.min().item())

        if min_pos < 0:
            raise ValueError(f"token_positions contains negative index {min_pos}")

        if max_pos >= self.max_seq_len:
            raise ValueError(
                f"token_positions max={max_pos} exceeds max_seq_len={self.max_seq_len}. "
                f"Increase max_seq_len."
            )

        # Split into even and odd parts: (..., seq_len, d_k/2)
        x_even = x[..., :, 0::2]
        x_odd = x[..., :, 1::2]

        # Select cos/sin for the given token positions: (..., seq_len, d_k/2)
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        # Apply rotation
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos

        # Interleave even/odd
        out = torch.empty_like(x)
        out[..., :, 0::2] = out_even
        out[..., :, 1::2] = out_odd
        # (..., seq_len, d_k/2, 2) -> (..., seq_len, d_k)
        # out = rearrange(
        #     torch.stack([out_even, out_odd], dim=-1),
        #     "... seq_len half_dim two -> ... seq_len (half_dim two)",
        # )

        return out