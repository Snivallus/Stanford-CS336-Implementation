import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from jaxtyping import Float, Int

from cs336_basics.Transformer.Basic_Building_Blocks import Embedding
from cs336_basics.Transformer.Basic_Building_Blocks import Linear
from cs336_basics.Transformer.Basic_Building_Blocks import RMSNorm
from cs336_basics.Transformer.Basic_Building_Blocks import TransformerBlock


@dataclass
class TransformerStats:
    num_parameters: int
    param_bytes: int
    flops_forward: int
    flops_breakdown: dict[str, int]

    def pretty(self) -> str:
        def fmt_bytes(x: int) -> str:
            if x < 1024:
                return f"{x} B"
            if x < 1024**2:
                return f"{x/1024:.3f} KB"
            if x < 1024**3:
                return f"{x/1024**2:.3f} MB"
            return f"{x/1024**3:.3f} GB"

        return (
            f"Trainable parameters: {self.num_parameters:,}\n"
            f"Parameter storage:    {fmt_bytes(self.param_bytes)}\n"
            f"FLOPs / forward:      {self.flops_forward:,}"
        )


class TransformerLM(nn.Module):
    """
    Transformer Language Model (causal decoder-only).

    This module implements a standard decoder-only Transformer:

        token_ids -> token_embedding -> TransformerBlocks -> RMSNorm -> output projection

    Notes:
        - Uses token embeddings only (positional information is injected via RoPE).
        - Uses pre-norm Transformer blocks.
        - Output is logits over vocabulary.
    """


    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: Optional[int] = None,
        rope_theta: Optional[float] = None,
        eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize the Transformer language model.

        Parameters:
            - vocab_size (int):
                Vocabulary size.

            - context_length (int):
                Maximum supported context length.

            - num_layers (int):
                Number of Transformer blocks.

            - d_model (int):
                Model embedding dimension.

            - num_heads (int):
                Number of attention heads.

            - d_ff (int | None):
                Hidden dimension of FFN.

            - rope_theta (float | None):
                RoPE Θ constant. If None, RoPE is disabled.

            - eps (float):
                RMSNorm epsilon.

            - device (torch.device | None):
                Device for parameters.

            - dtype (torch.dtype | None):
                Datatype for parameters.
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.token_embedding = Embedding(
            num_embeddings = vocab_size,
            embedding_dim = d_model,
            device = device,
            dtype = dtype,
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model = d_model,
                    num_heads = num_heads,
                    d_ff = d_ff,
                    rope_theta = rope_theta,
                    max_seq_len = context_length,
                    eps = eps,
                    device = device,
                    dtype = dtype,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = RMSNorm(
            d_model = d_model,
            eps = eps,
            device = device,
            dtype = dtype,
        )

        self.lm_head = Linear(
            in_features = d_model,
            out_features = vocab_size,
            device = device,
            dtype = dtype,
        )


    def forward(
        self,
        token_ids: Int[Tensor, "... seq_len"],
        token_positions: Optional[Int[Tensor, "... seq_len"]] = None,
    ) -> Float[Tensor, "... seq_len vocab_size"]:
        """
        Run the Transformer LM forward pass.

        Parameters:
            - token_ids (Tensor):
                Input token ids of shape (..., seq_len).

            - token_positions (Tensor | None):
                Optional token positions of shape (..., seq_len).
                If None, will default to [0, 1, ..., seq_len-1].

        Returns:
            - logits (Tensor):
                Output logits of shape (..., seq_len, vocab_size).
        """
        seq_len = token_ids.shape[-1]

        if seq_len > self.context_length:
            raise ValueError(
                f"seq_len = {seq_len} exceeds context_length = {self.context_length}"
            )

        # (..., seq_len) -> (..., seq_len, d_model)
        x = self.token_embedding(token_ids)  

        # (..., seq_len, d_model) -> (..., seq_len, d_model)
        for block in self.blocks:
            x = block(x, token_positions=token_positions)

        # (..., seq_len, d_model) -> (..., seq_len, d_model)
        x = self.final_norm(x)

        # (..., seq_len, d_model) -> (..., seq_len, vocab_size)
        logits = self.lm_head(x)

        return logits
    

    def statistics(
        self,
        batch_size: int,
        seq_len: int,
    ) -> TransformerStats:
        """
        Estimate:
        - number of trainable parameters + required storage
        - FLOPs per forward pass (dominant matmul terms)

        FLOPs counting rule:
            matmul (M x K) @ (K x N) costs about 2*M*K*N FLOPs (multiply + add).

        We count dominant matmuls:
        - per-layer:
            QKV projection
            attention scores QK^T
            attention weighted sum Attn*V
            output projection
            FFN SwiGLU projections (W1, W3, W2)
        - final lm head projection

        We ignore smaller operations (softmax, RMSNorm, RoPE rotation, masking).
        """
        if seq_len > self.context_length:
            raise ValueError(
                f"seq_len = {seq_len} exceeds context_length = {self.context_length}"
            )

        # -----------------------
        # Parameter count + bytes
        # -----------------------
        num_params = 0
        param_bytes = 0
        for p in self.parameters():
            if p.requires_grad:
                num_params += p.numel()
                param_bytes += p.numel() * p.element_size()

        # -----------------------
        # FLOPs estimation
        # -----------------------
        num_layers = self.num_layers
        d_model = self.d_model
        num_heads = self.num_heads
        head_dim = d_model // num_heads # d_k and d_v in multihead self-attention
        if self.d_ff is not None:
            d_ff = self.d_ff
        else:
            d_ff = self.blocks[0].d_ff
        vocab_size = self.vocab_size

        def matmul_flops(M: int, K: int, N: int) -> int:
            """FLOPs for (M x K) @ (K x N)."""
            return 2 * M * K * N

        # Compute FLOPs for ONE sequence (batch dimension excluded for now).
        flops_per_sequence = 0

        # ---- per layer FLOPs ----
        # Input per layer: (seq_len, d_model)

        # QKV projection:
        # (seq_len, d_model) @ (d_model, 3*d_model) -> (seq_len, 3*d_model)
        flops_qkv = matmul_flops(seq_len, d_model, 3 * d_model)

        # Attention scores QK^T:
        # for each head: (seq_len, head_dim) @ (head_dim, seq_len) -> (seq_len, seq_len)
        flops_qk = num_heads * matmul_flops(seq_len, head_dim, seq_len)

        # Attention-weighted sum:
        # for each head: (seq_len, seq_len) @ (seq_len, head_dim) -> (seq_len, head_dim)
        flops_av = num_heads * matmul_flops(seq_len, seq_len, head_dim)

        # Output projection:
        # (seq_len, d_model) @ (d_model, d_model) -> (seq_len, d_model)
        flops_out = matmul_flops(seq_len, d_model, d_model)

        # FFN SwiGLU:
        # w1: (seq_len, d_model) @ (d_model, d_ff) -> (seq_len, d_ff)
        # w3: (seq_len, d_model) @ (d_model, d_ff) -> (seq_len, d_ff)
        # w2: (seq_len, d_ff)    @ (d_ff, d_model) -> (seq_len, d_model)
        flops_ffn_w1 = matmul_flops(seq_len, d_model, d_ff)
        flops_ffn_w3 = matmul_flops(seq_len, d_model, d_ff)
        flops_ffn_w2 = matmul_flops(seq_len, d_ff, d_model)
        flops_ffn = flops_ffn_w1 + flops_ffn_w3 + flops_ffn_w2

        # Sum per-layer FLOPs:
        flops_per_layer = flops_qkv + flops_qk + flops_av + flops_out + flops_ffn
        flops_per_sequence += num_layers * flops_per_layer

        # Final LM head:
        # (seq_len, d_model) @ (d_model, vocab_size) -> (seq_len, vocab_size)
        flops_lm_head = matmul_flops(seq_len, d_model, vocab_size)
        flops_per_sequence += flops_lm_head

        # Multiply by batch size at the end
        flops_forward = batch_size * flops_per_sequence

        breakdown = {
            "layers.qkv_proj": batch_size * num_layers * flops_qkv,
            "layers.attn_scores(QK^T)": batch_size * num_layers * flops_qk,
            "layers.attn_weighted_sum(AV)": batch_size * num_layers * flops_av,
            "layers.output_proj": batch_size * num_layers * flops_out,
            "layers.ffn_w1": batch_size * num_layers * flops_ffn_w1,
            "layers.ffn_w3": batch_size * num_layers * flops_ffn_w3,
            "layers.ffn_w2": batch_size * num_layers * flops_ffn_w2,
            "lm_head": batch_size * flops_lm_head,
        }

        return TransformerStats(
            num_parameters = num_params,
            param_bytes = param_bytes,
            flops_forward = flops_forward,
            flops_breakdown = breakdown,
        )


def print_stats(name: str, stats: TransformerStats):
    print("=" * 80)
    print(f"Model: {name}")
    print("-" * 80)
    print(f"Trainable parameters: {stats.num_parameters:,}")
    print(f"Parameter memory (bytes): {stats.param_bytes:,}")
    print(f"Parameter memory (MB): {stats.param_bytes / (1024**2):.2f} MB")
    print(f"Parameter memory (GB): {stats.param_bytes / (1024**3):.2f} GB")
    print(f"Total FLOPs (forward): {stats.flops_forward:,}")

    print("\nFLOPs breakdown:")
    total = stats.flops_forward
    for k, v in sorted(stats.flops_breakdown.items(), key=lambda x: -x[1]):
        print(f"  {k:35s} {v:>20,}   ({100.0 * v / total:6.2f}%)")
    print("=" * 80)
    print()


if __name__ == "__main__":
    device = torch.device("cpu")
    dtype = torch.float32

    vocab_size = 50257

    configs = {
        "GPT2-small": dict(context_length=1024, num_layers=12, d_model=768,  num_heads=12, d_ff=3072),
        "GPT2-medium": dict(context_length=1024, num_layers=24, d_model=1024, num_heads=16, d_ff=4096),
        "GPT2-large": dict(context_length=1024, num_layers=36, d_model=1280, num_heads=20, d_ff=5120),
        "GPT2-XL": dict(context_length=1024, num_layers=48, d_model=1600, num_heads=25, d_ff=6400),
        "GPT2-XL (longer context)": dict(context_length=16384, num_layers=48, d_model=1600, num_heads=25, d_ff=6400)
    }

    rope_theta = 10000.0  # typical GPT-style RoPE theta

    batch_size = 1

    for model_name, cfg in configs.items():
        model = TransformerLM(
            vocab_size = vocab_size,
            context_length = cfg["context_length"],
            d_model = cfg["d_model"],
            num_layers = cfg["num_layers"],
            num_heads = cfg["num_heads"],
            d_ff = cfg["d_ff"],
            rope_theta = rope_theta,
            device = device,
            dtype = dtype,
        )

        stats = model.statistics(batch_size=batch_size, seq_len=cfg["context_length"])
        print_stats(model_name, stats)