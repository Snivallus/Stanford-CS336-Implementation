import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from jaxtyping import Float

from cs336_basics.Transformer.Basic_Building_Blocks import Linear


class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network (FFN) module.

    This module implements the SwiGLU transformation:

        FFN(x) = W2( SiLU(W1 x) ⊙ (W3 x) )

    where:
        - x has last dimension d_model
        - W1, W3 have shape (d_ff, d_model)
        - W2 has shape (d_model, d_ff)
        - ⊙ is elementwise (Hadamard) product
        - SiLU(z) = z * sigmoid(z)

    Notes:
        - This implementation omits bias terms.
        - The hidden dimension d_ff is chosen as 8 * d_model / 3, rounded down
          to a multiple of 64 for hardware efficiency.
        - Uses custom Linear modules (not nn.Linear).
    """


    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize the SwiGLU feed-forward network.

        Parameters:
            - d_model (int):
                Input and output dimensionality of the module.

            - d_ff (int | None):
                Hidden dimension used in the SwiGLU block.

            - device (torch.device | None):
                Device on which the parameters will be allocated.

            - dtype (torch.dtype | None):
                Datatype of the parameters.

        Attributes:
            - w1 (Linear):
                Linear map from d_model -> d_ff.
            - w2 (Linear):
                Linear map from d_ff -> d_model.
            - w3 (Linear):
                Linear map from d_model -> d_ff.
        """
        super().__init__()

        self.d_model = d_model

        if d_ff is None:
            # Canonical dimension: d_ff = 8 * d_model / 3
            d_ff = int(math.floor((8 * d_model) / 3))

            # Round down to a multiple of 64 (minimum 64)
            d_ff = max(64, (d_ff // 64) * 64)
        self.d_ff = d_ff

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)


    def forward(
        self, 
        x: Float[Tensor, "... d_model"],
    ) -> Float[Tensor, "... d_model"]:
        """
        Apply the SwiGLU feed-forward network.

        Parameters:
            - x (Tensor):
                Input tensor of shape (..., d_model), typically
                (batch_size, sequence_length, d_model).

        Returns:
            - output (Tensor):
                Output tensor of shape (..., d_model).

        Notes:
            - SiLU is implemented explicitly using torch.sigmoid:

                SiLU(z) = z * sigmoid(z)
        """
        z1: Float[Tensor, "... d_ff"] = self.w1(x)
        z3: Float[Tensor, "... d_ff"] = self.w3(x)

        silu_z1: Float[Tensor, "... d_ff"] = z1 * torch.sigmoid(z1)
        gated: Float[Tensor, "... d_ff"] = silu_z1 * z3

        out: Float[Tensor, "... d_model"] = self.w2(gated)
        return out