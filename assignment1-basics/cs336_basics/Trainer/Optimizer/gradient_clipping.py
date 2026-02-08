import torch
from torch import Tensor
from typing import Iterable


def gradient_clipping(
    params: Iterable[Tensor], 
    max_l2_norm: float, 
    eps: float = 1e-6
) -> Tensor:
    """
    Clip gradients in-place to enforce a maximum global L2 norm.

    Args:
        - params: Iterable of parameters (each may have .grad).
        - max_l2_norm: Maximum allowed global L2 norm of the gradients.
        - eps: Small constant for numerical stability.

    Returns:
        - total_norm: Scalar tensor, the original global L2 norm before clipping.
    """
    if max_l2_norm < 0:
        raise ValueError(f"max_norm must be non-negative, got {max_l2_norm}")

    # Compute global L2 norm of all gradients.
    total_sq_norm = torch.zeros((), device="cpu")
    grads = []

    for p in params:
        if p.grad is None:
            continue
        g = p.grad
        grads.append(g)
        total_sq_norm = total_sq_norm.to(g.device)
        total_sq_norm += torch.sum(g * g)

    total_norm = torch.sqrt(total_sq_norm)

    # Clip gradients if needed.
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for g in grads:
            g.mul_(scale)

    return total_norm