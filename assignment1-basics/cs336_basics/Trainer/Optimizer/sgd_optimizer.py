from collections.abc import Callable
from typing import List, Optional
import math
import torch


class SGD(torch.optim.Optimizer):
    def __init__(
        self, 
        params: List[torch.nn.Parameter],
        lr: float = 1e-3
    ) -> None:
        """
        SGD with learning rate schedule lr / sqrt(t+1).

        Args:
            - params: iterable of parameters or parameter groups
            - lr: initial learning rate alpha_0
        """
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {"lr": lr}
        super().__init__(params, defaults)


    def step(
        self, 
        closure: Optional[Callable] = None
    ) -> Optional[float]:
        """
        Perform one optimization step.

        Args:
            - closure: a closure that reevaluates the model and returns the loss.
        """
        loss = None if closure is None else closure()

        with torch.no_grad():
            for group in self.param_groups:
                lr = group["lr"]

                for p in group["params"]:
                    if p.grad is None:
                        continue

                    state = self.state[p]
                    t = state.get("t", 0)

                    grad = p.grad

                    # alpha_t = lr / sqrt(t+1)
                    step_size = lr / math.sqrt(t + 1)
                    p.add_(grad, alpha=-step_size)

                    state["t"] = t + 1

        return loss


if __name__ == "__main__":
    torch.manual_seed(51)

    for lr in [1e1, 1e2, 1e3]:
        print(f"\n==== Running SGD with lr = {lr} ====")

        weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
        opt = SGD([weights], lr=lr)

        for t in range(10):
            opt.zero_grad()
            loss = (weights ** 2).mean()
            print(f"iter {t:2d}: loss = {loss.item():.6f}")
            loss.backward()
            opt.step()