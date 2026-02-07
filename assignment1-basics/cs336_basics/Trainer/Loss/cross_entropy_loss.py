import torch
from torch import Tensor
from jaxtyping import Float, Int
from typing import Tuple


def cross_entropy_loss(
    logits: Float[Tensor, "... vocab_size"],
    target: Int[Tensor, "..."],
) -> Tuple[Tensor, Tensor]:
    """
    Compute the mean cross-entropy loss and perplexity from unnormalized logits.

    Args:
        - logits: Tensor of shape (..., vocab_size), unnormalized model outputs.
        - target: Tensor of shape (...), integer class indices in [0, vocab_size).

    Returns:
        - loss_mean: Scalar tensor, mean cross-entropy over all batch dimensions.
        - perplexity: Scalar tensor, exp(loss_mean).
    """
    # Compute in float32 for numerical stability (important for fp16/bf16 logits).
    logits = logits.float()

    # Shift logits by their maximum value to improve numerical stability.
    logits_shifted = logits - logits.max(dim=-1, keepdim=True).values

    # logsumexp along vocab dimension.
    logsumexp = torch.logsumexp(logits_shifted, dim=-1)

    # Extract the shifted logit corresponding to the target class.
    target_logit = torch.gather(logits_shifted, dim=-1, index=target.unsqueeze(-1)).squeeze(-1)

    # CE = -log softmax(target) = -target_logit + logsumexp
    loss = -target_logit + logsumexp

    # Mean loss across all batch dimensions.
    loss_mean = loss.mean()

    # Perplexity = exp(mean loss).
    perplexity = torch.exp(loss_mean)

    return loss_mean, perplexity