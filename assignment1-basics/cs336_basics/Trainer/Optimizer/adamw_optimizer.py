from collections.abc import Callable
from typing import Optional, Tuple
import math
import torch


class AdamW(torch.optim.Optimizer):
    """
    AdamW optimizer with bias correction and decoupled weight decay.

    Reference:
        - [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
    """


    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        """
        AdamW optimizer with bias correction and decoupled weight decay.

        Args:
            - params: Iterable of parameters or parameter groups.
            - lr: Learning rate alpha.
            - betas: Tuple (beta1, beta2) controlling moment decay rates.
            - eps: Small constant epsilon for numerical stability.
            - weight_decay: Weight decay rate lambda.
        """
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        beta1, beta2 = betas
        if not (0.0 <= beta1 < 1.0):
            raise ValueError(f"Invalid beta1: {beta1}")
        if not (0.0 <= beta2 < 1.0):
            raise ValueError(f"Invalid beta2: {beta2}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)


    def step(
        self, 
        closure: Optional[Callable] = None
    ):
        """
        Perform one AdamW optimization step.

        Args:
            - closure: a closure that reevaluates the model and returns the loss.
        """
        loss = None if closure is None else closure()

        with torch.no_grad():
            for group in self.param_groups:
                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]

                for p in group["params"]:
                    if p.grad is None:
                        continue

                    grad = p.grad

                    state = self.state[p]
                    t = state.get("t", 0) + 1  # t starts from 1

                    # Initialize state (first time we see this parameter).
                    if "m" not in state:
                        state["m"] = torch.zeros_like(p)
                        state["v"] = torch.zeros_like(p)

                    m = state["m"]
                    v = state["v"]

                    # Update biased first and second moment estimates.
                    m.mul_(beta1).add_(grad, alpha=(1 - beta1))
                    v.mul_(beta2).addcmul_(grad, grad, value=(1 - beta2))

                    # Bias-corrected learning rate.
                    alpha_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)

                    # Parameter update.
                    p.addcdiv_(m, v.sqrt().add_(eps), value=-alpha_t)

                    # Decoupled weight decay: theta <- (1 - lr * lambda) * theta
                    if weight_decay != 0.0:
                        p.mul_(1 - lr * weight_decay)

                    # Save updated step count.
                    state["t"] = t

        return loss


if __name__ == "__main__":
    torch.manual_seed(0)

    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = AdamW([weights], lr=1, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-2)

    for t in range(10):
        opt.zero_grad()
        loss = (weights ** 2).mean()
        print(f"iter {t:2d}: loss = {loss.item():.6f}")
        loss.backward()
        opt.step()