import os
import torch
from torch import nn
from torch.optim import Optimizer
from typing import BinaryIO, IO, Union, Optional, Dict, Any


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    iteration: int,
    out: Union[str, os.PathLike, BinaryIO, IO[bytes]],
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a training checkpoint containing model state, optimizer state, iteration,
    and optionally trainer args/config.

    Args:
        - model: PyTorch model (nn.Module).
        - optimizer: PyTorch optimizer (torch.optim.Optimizer).
        - iteration: Current training iteration to resume from.
        - out: Output path or file-like object for torch.save().
        - config: Optional dict of training config/args.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }

    if config is not None:
        checkpoint["config"] = config

    torch.save(checkpoint, out)