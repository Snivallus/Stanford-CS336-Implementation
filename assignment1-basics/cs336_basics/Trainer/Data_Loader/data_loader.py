import os
import numpy as np
import torch
from torch import Tensor
from typing import Union, Tuple


def data_loader(
    x: Union[str, np.ndarray],
    batch_size: int,
    context_length: int,
    device: str,
) -> Tuple[Tensor, Tensor]:
    """
    Sample a batch of (input, target) sequences from a long token stream.

    Args:
        - x: Numpy array of token IDs (shape: [N]) or path to a .npy file.
        - batch_size: Number of sequences in the batch (B).
        - context_length: Length of each sequence (m).
        - device: PyTorch device string (e.g. "cpu", "cuda").

    Returns:
        - inputs: Tensor of shape (batch_size, context_length) on `device`.
        - targets: Tensor of shape (batch_size, context_length) on `device`.
    """
    # load x if it's a file path
    # use memory-mapped loading to avoid reading the entire array into RAM
    if isinstance(x, str):
        if not os.path.isfile(x):
            raise ValueError(f"x is a string but not a valid file path: {x}")

        if x.endswith(".npy"):
            # memory-mapped loading to avoid reading the full array into RAM
            x = np.load(x, mmap_mode="r") # requires file created via np.save()
        else:
            raise ValueError(
                f"Unsupported file format for x: {x}. Expected a .npy file.\n"
                "If it is a raw binary file, use np.memmap() to load it (with dtype/shape) "
                "and pass the resulting array directly."
            )

    elif isinstance(x, np.ndarray):
        pass

    else:
        raise ValueError(f"x must be a numpy array or a valid .npy file path, got {type(x)}")

    if x.ndim != 1:
        raise ValueError(f"x must be a 1D array of token IDs, got shape {x.shape}")

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    if context_length <= 0:
        raise ValueError(f"context_length must be positive, got {context_length}")

    if len(x) < context_length + 1:
        raise ValueError(
            f"Token stream too short: need at least {context_length+1} tokens, got {len(x)}"
        )

    # sample starting indices uniformly
    max_start = len(x) - context_length - 1
    starts = np.random.randint(0, max_start + 1, size=batch_size)

    # build batch
    inputs = np.stack([x[s : s + context_length] for s in starts])
    targets = np.stack([x[s + 1 : s + 1 + context_length] for s in starts])

    # convert to torch tensors on the requested device
    inputs = torch.tensor(inputs, dtype=torch.int64, device=device)
    targets = torch.tensor(targets, dtype=torch.int64, device=device)

    return inputs, targets