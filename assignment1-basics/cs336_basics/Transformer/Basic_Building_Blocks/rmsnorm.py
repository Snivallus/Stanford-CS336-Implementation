import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from jaxtyping import Float


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Given an input tensor x, RMSNorm rescales activations along the final
    dimension (d_model) as:

        RMSNorm(x_i) = (x_i / RMS(x)) * g_i

    where g is a learnable scale parameter and:

        RMS(x) = sqrt(mean(x^2) + eps)

    Notes:
        - Normalization is performed over the last dimension only.
        - Input is upcast to float32 during normalization to prevent overflow.
        - The output is cast back to the original dtype.
    """


    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize the RMSNorm module.

        Parameters:
            - d_model (int):
                Dimension of the last axis to normalize over.

            - eps (float):
                Small constant added for numerical stability.

            - device (torch.device | None):
                Device on which the learnable scale parameter will be allocated.

            - dtype (torch.dtype | None):
                Datatype of the learnable scale parameter.

        Attributes:
            - weight (nn.Parameter):
                Learnable scale parameter g of shape (d_model,).
        """
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        # "gain" parameters
        self.weight = nn.Parameter(
            torch.ones(d_model, device = device, dtype = dtype)
        )


    def forward(
        self, 
        x: Float[Tensor, "... d_model"],
    ) -> Float[Tensor, "... d_model"]:
        """
        Apply RMSNorm to the input tensor.

        Parameters:
            - x (Tensor):
                Input tensor of shape (..., d_model), typically (batch, seq_len, d_model).

        Returns:
            - output (Tensor):
                Tensor of the same shape as x after RMS normalization and scaling.

        Notes:
            - The input is upcast to float32 to avoid overflow during squaring.
            - The output is cast back to the original dtype.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        y = (x / rms) * self.weight

        return y.to(in_dtype)