import os
import torch
from torch import nn
from torch.optim import Optimizer
from typing import BinaryIO, IO, Union, Dict, Any, Tuple


def load_checkpoint(
    src: Union[str, os.PathLike, BinaryIO, IO[bytes]],
    model: nn.Module,
    optimizer: Optimizer,
    return_config: bool = False,
) -> Union[int, Tuple[int, Dict[str, Any]]]:
    """
    Load a training checkpoint and restore model + optimizer states.

    Args:
        - src: Checkpoint path or file-like object for torch.load().
        - model: Model to restore into.
        - optimizer: Optimizer to restore into.
        - return_config: If True, also return config dict stored in checkpoint.

    Returns:
        - iteration (int) if return_config=False
        - (iteration, config_dict) if return_config=True
    """
    checkpoint = torch.load(src, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    iteration = checkpoint["iteration"]

    if not return_config:
        return iteration

    config = checkpoint.get("config", {})
    if config is None:
        config = {}

    assert isinstance(config, dict), "checkpoint['config'] must be a dict"
    return iteration, config


def load_checkpoint_config(
    src: Union[str, os.PathLike, BinaryIO, IO[bytes]],
) -> Dict[str, Any]:
    """
    Load only the config dict from a checkpoint.

    Args:
        - src: Checkpoint path or file-like object for torch.load().

    Returns:
        - config dict (empty dict if missing or None)
    """
    checkpoint = torch.load(src, map_location="cpu")

    config = checkpoint.get("config")
    if config is None:
        config = {}

    assert isinstance(config, dict), "checkpoint['config'] must be a dict"
    return config


def load_checkpoint_model(
    src: Union[str, os.PathLike, BinaryIO, IO[bytes]],
    model: nn.Module,
    strict: bool = True,
) -> None:
    """
    Load only the model weights from a checkpoint.

    Args:
        - src: Checkpoint path or file-like object for torch.load().
        - model: Model to restore into.
        - strict: Whether to strictly enforce that the keys match.
    """
    checkpoint = torch.load(src, map_location="cpu")

    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise KeyError("checkpoint does not contain 'model_state_dict'")

    model.load_state_dict(state_dict, strict=strict)