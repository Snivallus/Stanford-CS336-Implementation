import os
import time
import argparse
import numpy as np
import torch

from cs336_basics.Transformer import TransformerLM
from cs336_basics.Trainer.Loss import cross_entropy_loss
from cs336_basics.Trainer.Optimizer import AdamW, gradient_clipping
from cs336_basics.Trainer.Scheduler import cosine_annealing_scheduler
from cs336_basics.Trainer.Data_Loader import data_loader
from cs336_basics.Trainer.Checkpointing import save_checkpoint, load_checkpoint


def evaluate(
    model: torch.nn.Module,
    val_tokens: str,
    batch_size: int,
    context_length: int,
    device: str,
    num_batches: int,
) -> tuple[float, float]:
    """
    Evaluate average loss and perplexity on validation set.

    Args:
        - model: TransformerLM.
        - val_tokens: Path to validation .npy file.
        - batch_size: Batch size.
        - context_length: Context length.
        - device: Device string.
        - num_batches: Number of validation batches.

    Returns:
        - val_loss: Mean validation loss.
        - val_ppl: Mean validation perplexity.
    """
    model.eval()
    losses = []

    with torch.no_grad():
        for _ in range(num_batches):
            x, y = data_loader(
                x = val_tokens,
                batch_size = batch_size,
                context_length = context_length,
                device = device,
            )
            logits = model(x)
            loss = cross_entropy_loss(logits, y)
            losses.append(loss.item())

    val_loss = float(np.mean(losses))
    val_ppl = float(np.exp(min(val_loss, 20))) # cap loss to prevent overflow in exp
    model.train()
    return val_loss, val_ppl


def main():
    parser = argparse.ArgumentParser()

    # -----------------------
    # Data / training config
    # -----------------------
    parser.add_argument("--train_tokens", type=str, required=True, help="Path to training .npy file")
    parser.add_argument("--val_tokens", type=str, required=True, help="Path to validation .npy file")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training")
    parser.add_argument("--dtype", type=str, default="float32", help="Data type: float32, float16, bfloat16")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--context_length", type=int, default=256, help="Context length")
    parser.add_argument("--max_iters", type=int, default=1000, help="Maximum training iterations")

    # -----------------------
    # Model config
    # -----------------------
    parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of Transformer layers")
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=2048, help="Feedforward network dimension")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE theta parameter")

    # -----------------------
    # Optimizer config
    # -----------------------
    parser.add_argument("--lr_max", type=float, default=3e-4, help="Maximum learning rate")
    parser.add_argument("--lr_min", type=float, default=3e-5, help="Minimum learning rate")
    parser.add_argument("--betas1", type=float, default=0.9, help="Beta1 for AdamW")
    parser.add_argument("--betas2", type=float, default=0.95, help="Beta2 for AdamW")
    parser.add_argument("--eps", type=float, default=1e-8, help="Epsilon for AdamW")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay for AdamW")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Max gradient norm for clipping")

    # -----------------------
    # Scheduler config
    # -----------------------
    parser.add_argument("--warmup_iters", type=int, default=100, help="Number of warmup iterations")
    parser.add_argument("--cosine_iters", type=int, default=1000, help="Number of cosine annealing iterations")

    # -----------------------
    # Logging / checkpointing
    # -----------------------
    parser.add_argument("--log_every", type=int, default=10, help="Log every N iterations")
    parser.add_argument("--eval_every", type=int, default=100, help="Evaluate every N iterations")
    parser.add_argument("--eval_batches", type=int, default=20, help="Number of batches for evaluation")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to save/load checkpoint")
    parser.add_argument("--resume", action="store_true", help="Whether to resume from checkpoint if available")

    # -----------------------
    # Early stopping
    # -----------------------
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0)

    # -----------------------
    # Random seed
    # -----------------------
    parser.add_argument("--seed", type=int, default=51, help="Random seed")

    args = parser.parse_args()

    # -----------------------
    # Set random seed
    # -----------------------
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Make cuDNN deterministic (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # -----------------------
    # Setup dtype
    # -----------------------
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if args.dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype: {args.dtype}")
    dtype = dtype_map[args.dtype]

    # -----------------------
    # Model
    # -----------------------
    model = TransformerLM(
        vocab_size = args.vocab_size,
        context_length = args.context_length,
        num_layers = args.num_layers,
        d_model = args.d_model,
        num_heads = args.num_heads,
        d_ff = args.d_ff,
        rope_theta = args.rope_theta,
        device = args.device,
        dtype = dtype,
    )

    # -----------------------
    # Optimizer
    # -----------------------
    optimizer = AdamW(
        model.parameters(),
        lr = args.lr_max,
        betas = (args.betas1, args.betas2),
        eps = args.eps,
        weight_decay = args.weight_decay,
    )

    # -----------------------
    # Resume checkpoint
    # -----------------------
    start_iter = 0
    if args.resume:
        if args.checkpoint_path is None or not os.path.isfile(args.checkpoint_path):
            raise ValueError("resume=True requires a valid --checkpoint_path")

        start_iter = load_checkpoint(args.checkpoint_path, model, optimizer)
        print(f"[Resume] Loaded checkpoint from {args.checkpoint_path} at iter = {start_iter}")

    # -----------------------
    # Initialize early stopping
    # -----------------------
    best_val_loss = float("inf")
    best_iter = -1
    bad_evals = 0

    # evaluate once at start (especially if resuming)
    val_loss, val_ppl = evaluate(
        model = model,
        val_tokens = args.val_tokens,
        batch_size = args.batch_size,
        context_length = args.context_length,
        device = args.device,
        num_batches = args.eval_batches,
    )
    best_val_loss = val_loss
    best_iter = start_iter
    if args.resume:
        print(f"[Resume] initial val_loss = {val_loss:.4f} ppl = {val_ppl:.2f}")
    else:
        print(f"[Start] initial val_loss = {val_loss:.4f} ppl = {val_ppl:.2f}")
        if args.checkpoint_path is not None:
            save_checkpoint(model, optimizer, start_iter, args.checkpoint_path)
            print(f"[checkpoint] saved initial checkpoint to {args.checkpoint_path}")


    # -----------------------
    # Training loop
    # -----------------------
    model.train()
    t0 = time.time()

    for iter in range(start_iter + 1, args.max_iters):
        # scheduler updates optimizer lr
        lr_t = cosine_annealing_scheduler(
            t = iter,
            alpha_max = args.lr_max,
            alpha_min = args.lr_min,
            T_w = args.warmup_iters,
            T_c = args.cosine_iters,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr_t

        # sample batch
        x, y = data_loader(
            x = args.train_tokens,
            batch_size = args.batch_size,
            context_length = args.context_length,
            device = args.device,
        )

        # forward + loss
        logits = model(x)
        loss = cross_entropy_loss(logits, y)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient clipping
        grad_norm = gradient_clipping(
            params = model.parameters(), 
            max_l2_norm = args.grad_clip
        )

        # step
        optimizer.step()

        # logging
        if iter % args.log_every == 0:
            dt = time.time() - t0
            ppl = float(torch.exp(loss.detach().float()).cpu().item())
            print(
                f"[iter {iter:6d}] "
                f"loss = {loss.item():.4f} ppl = {ppl:.2f} "
                f"lr = {lr_t:.6e} grad_norm = {grad_norm.item():.4f} "
                f"time = {dt:.2f} sec"
            )

        # -----------------------------------------------
        # Validation + early stopping + best checkpoint
        # -----------------------------------------------
        if iter % args.eval_every == 0:
            val_loss, val_ppl = evaluate(
                model = model,
                val_tokens = args.val_tokens,
                batch_size = args.batch_size,
                context_length = args.context_length,
                device = args.device,
                num_batches = args.eval_batches,
            )
            print(f"[val] loss = {val_loss:.4f} ppl = {val_ppl:.2f}")

            improved = val_loss < (best_val_loss - args.early_stop_min_delta)

            if improved:
                best_val_loss = val_loss
                best_iter = iter
                bad_evals = 0
                print(f"[best] new best val_loss = {best_val_loss:.4f} at iter = {best_iter}")

                if args.checkpoint_path is not None:
                    save_checkpoint(model, optimizer, iter, args.checkpoint_path)
                    print(f"[checkpoint] saved BEST checkpoint to {args.checkpoint_path}")

            else:
                bad_evals += 1
                print(
                    f"[early_stop] no improvement ({bad_evals}/{args.early_stop_patience}) "
                    f"best_val_loss = {best_val_loss:.4f} at iter = {best_iter}"
                )

                if bad_evals >= args.early_stop_patience:
                    print(
                        f"[early_stop] stopping training: "
                        f"best_val_loss = {best_val_loss:.4f} at iter = {best_iter}"
                    )
                    break


    print("Training finished.")


if __name__ == "__main__":
    main()