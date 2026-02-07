import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from jaxtyping import Float


class Embedding(nn.Module):
    """
    A embedding module that maps integer token IDs to embedding vectors.

    Given token IDs of shape (batch, seq_len), this module returns
    embedding vectors of shape (batch, seq_len, embedding_dim).

    Notes:
        - The embedding matrix is stored as a learnable parameter of shape
          (num_embeddings, embedding_dim), where embedding_dim is the final dimension.
        - The embedding matrix is initialized using a truncated normal distribution:

            W_ij ~ N(0, 1), truncated to [-3, 3]
    """


    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize an embedding layer.

        Parameters:
            - num_embeddings (int):
                Size of the vocabulary, i.e., the number of unique token IDs.

            - embedding_dim (int):
                Dimension of the embedding vectors (d_model).

            - device (torch.device | None):
                Device on which the embedding parameter will be allocated.
                If None, uses PyTorch default device behavior.

            - dtype (torch.dtype | None):
                Datatype of the embedding parameter.
                If None, uses PyTorch default dtype behavior.

        Attributes:
            - weight (nn.Parameter):
                Learnable embedding matrix of shape (num_embeddings, embedding_dim).
        """
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Allocate embedding matrix
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )

        # Truncated normal initialization: mean=0, std=1, truncated to [-3, 3]
        nn.init.trunc_normal_(
            self.weight, 
            mean = 0.0, 
            std = 1.0, 
            a = -3.0, 
            b = 3.0
        )


    def forward(
        self, 
        token_ids: Float[Tensor, "... seq_len"],
    ) -> Float[Tensor, "... seq_len embedding_dim"]:
        """
        Lookup embedding vectors for the given token IDs.

        Parameters:
            - token_ids (Tensor):
                Tensor of integer token IDs with shape (batch, seq_len).

        Returns:
            - output (Tensor):
                Tensor of embedding vectors with shape
                (batch, seq_len, embedding_dim).

        Notes:
            - Uses simple indexing into the embedding matrix to select vectors.
            - Does NOT use nn.Embedding or nn.functional.embedding.
        """
        return self.weight[token_ids]