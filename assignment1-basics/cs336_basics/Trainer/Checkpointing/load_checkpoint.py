import os
import torch
from torch import nn
from torch.optim import Optimizer
from typing import BinaryIO, IO, Union


def load_checkpoint(
    src: Union[str, os.PathLike, BinaryIO, IO[bytes]],
    model: nn.Module,
    optimizer: Optimizer,
) -> int:
    """
    Load a training checkpoint and restore model + optimizer states.

    Args:
        - src: Checkpoint path or file-like object for torch.load().
        - model: Model to restore into.
        - optimizer: Optimizer to restore into.

    Returns:
        - iteration: Training iteration stored in the checkpoint.
    """
    checkpoint = torch.load(src, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint["iteration"]