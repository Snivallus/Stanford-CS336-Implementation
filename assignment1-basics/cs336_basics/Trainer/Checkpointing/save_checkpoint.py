import os
import torch
from torch import nn
from torch.optim import Optimizer
from typing import BinaryIO, IO, Union


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    iteration: int,
    out: Union[str, os.PathLike, BinaryIO, IO[bytes]],
) -> None:
    """
    Save a training checkpoint containing model state, optimizer state, and iteration.

    Args:
        - model: PyTorch model (nn.Module).
        - optimizer: PyTorch optimizer (torch.optim.Optimizer).
        - iteration: Current training iteration to resume from.
        - out: Output path or file-like object for torch.save().
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)



