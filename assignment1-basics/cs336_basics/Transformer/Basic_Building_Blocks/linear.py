import math
import torch
import torch.nn as nn
from torch import Tensor
from einops import einsum
from typing import Optional
from jaxtyping import Float


class Linear(nn.Module):
    """
    A bias-free linear transformation module implementing:

        y = W x

    where:
        - x has last dimension `in_features`
        - W is a learnable weight matrix of shape (out_features, in_features)
        - y has last dimension `out_features`

    This module mirrors the interface of `torch.nn.Linear` except that it does
    NOT include a bias term.

    Notes:
        - The weight is stored as W (not W^T) for memory ordering reasons.
        - The weight is initialized using a truncated normal distribution:

            W_ij ~ N(0, sigma^2), truncated to [-3*sigma, 3*sigma]

          where:
            sigma^2 = 2 / (in_features + out_features)
    """


    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize a bias-free linear layer.

        Parameters:
            - in_features (int):
                Final dimension of the input tensor x. The forward pass expects
                input tensors of shape (..., in_features).

            - out_features (int):
                Final dimension of the output tensor y. The forward pass produces
                output tensors of shape (..., out_features).

            - device (torch.device | None):
                Device on which the weight parameter will be allocated.
                If None, uses PyTorch default device behavior.

            - dtype (torch.dtype | None):
                Datatype of the weight parameter.
                If None, uses PyTorch default dtype behavior.

        Attributes:
            - weight (nn.Parameter):
                Learnable parameter W of shape (out_features, in_features).
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Allocate W directly in the correct shape (out_features, in_features)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )

        # Truncated normal initialization:
        std = math.sqrt(2.0 / (in_features + out_features)) # sigma^2 = 2 / (din + dout)
        nn.init.trunc_normal_(
            self.weight,
            mean = 0.0,
            std = std,
            a = -3.0 * std,
            b = 3.0 * std,
        )


    def forward(
        self, 
        x: Float[Tensor, "... d_in"],
    ) -> Float[Tensor, "... d_out"]:
        """
        Apply the linear transformation y = W x.

        Parameters:
            - x (Tensor):
                Input tensor of shape (..., in_features).
                The leading dimensions (batch dimensions) may be arbitrary.

        Returns:
            - output (Tensor):
                Output tensor of shape (..., out_features), computed as:

                    y[..., j] = sum_i W[j, i] * x[..., i]

        Notes:
            - This implementation uses `einops.einsum` to explicitly represent the
              contraction over the input feature dimension.
        """
        #                                  x       weight   ->     y
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")