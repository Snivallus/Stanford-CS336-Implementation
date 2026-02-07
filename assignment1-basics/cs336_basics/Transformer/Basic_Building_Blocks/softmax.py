import torch
from torch import Tensor
from jaxtyping import Float


def softmax(
    x: Float[Tensor, "..."], 
    dim: int
) -> Float[Tensor, "..."]:
    """
    Apply the softmax operation over a specified dimension of a tensor.

    Softmax is defined as:

        softmax(x_i) = exp(x_i) / sum_j exp(x_j)

    computed along the specified dimension `dim`.

    This function uses the standard numerical stability trick of subtracting
    the maximum value along the softmax dimension before exponentiation.

    Parameters:
        - x (Tensor):
            Input tensor of arbitrary shape.

        - dim (int):
            Dimension along which to apply the softmax normalization.

    Returns:
        - output (Tensor):
            Tensor of the same shape as `x`, where values along dimension `dim`
            form a normalized probability distribution (sum to 1).
    """
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_stable = x - x_max

    exp_x = torch.exp(x_stable)
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)

    return exp_x / sum_exp_x