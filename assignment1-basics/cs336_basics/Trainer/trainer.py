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
from cs336_basics.Trainer.Checkpointing import save_checkpoint, load_checkpoint_config, load_checkpoint


class Trainer:
    def __init__(self):
        parser = argparse.ArgumentParser()

        # -----------------------
        # Data / training config
        # -----------------------
        parser.add_argument("--train_tokens", type=str, required=True, help="Path to training .npy file")
        parser.add_argument("--valid_tokens", type=str, required=True, help="Path to validation .npy file")
        parser.add_argument("--device", type=str, default="cuda", help="Device to use for training")
        parser.add_argument("--dtype", type=str, default="float32", help="Data type: float32, float16, bfloat16", choices=["float32", "float16", "bfloat16"])
        parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
        parser.add_argument("--context_length", type=int, default=256, help="Context length")
        parser.add_argument("--max_steps", type=int, default=1000, help="Maximum training iterations")

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
        parser.add_argument("--warmup_steps", type=int, default=100, help="Number of warmup iterations")
        parser.add_argument("--cosine_steps", type=int, default=1000, help="Number of cosine annealing iterations")

        # -----------------------
        # Logging / checkpointing
        # -----------------------
        parser.add_argument("--log_every", type=int, default=10, help="Log every N iterations")
        parser.add_argument("--eval_every", type=int, default=100, help="Evaluate every N iterations")
        parser.add_argument("--eval_batches", type=int, default=20, help="Number of batches for evaluation")
        parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to save/load checkpoint")
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

        # -----------------------
        # Parse arguments
        # -----------------------
        self.args = parser.parse_args()

        # -----------------------
        # Setup checkpoint directory
        # -----------------------
        ckpt_dir = os.path.dirname(self.args.checkpoint_path)
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)

        # -----------------------------------------
        # Resume args from checkpoint if available
        # -----------------------------------------
        if self.args.resume:
            if not os.path.isfile(self.args.checkpoint_path):
                raise ValueError("--resume requires an existing checkpoint_path")

            saved_cfg = load_checkpoint_config(
                src = self.args.checkpoint_path,
            )

            # Keep CLI overrides
            cli_overrides = {
                "device": self.args.device,
                "checkpoint_path": self.args.checkpoint_path,
                "resume": self.args.resume,
                "train_tokens": self.args.train_tokens,
                "valid_tokens": self.args.valid_tokens,
            }

            # Warn if dataset path differs
            if "train_tokens" in saved_cfg and saved_cfg["train_tokens"] != cli_overrides["train_tokens"]:
                print(f"[Warning] train_tokens mismatch: ckpt = {saved_cfg['train_tokens']} cli = {self.args.train_tokens}")

            if "valid_tokens" in saved_cfg and saved_cfg["valid_tokens"] != cli_overrides["valid_tokens"]:
                print(f"[Warning] valid_tokens mismatch: ckpt = {saved_cfg['valid_tokens']} cli = {self.args.valid_tokens}")

            # Replace args with saved config
            for k, v in saved_cfg.items():
                if hasattr(self.args, k):
                    setattr(self.args, k, v)

            # Re-apply CLI overrides
            for k, v in cli_overrides.items():
                setattr(self.args, k, v)

        # -------------------------------------------
        # Print final config (after resume override)
        # -------------------------------------------
        print("\n========== Training Config (Final) ==========")
        for k, v in sorted(vars(self.args).items()):
            print(f"{k:25s}: {v}")
        print("=============================================\n")

        # -----------------------
        # Set random seed
        # -----------------------
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.args.seed)
            torch.cuda.manual_seed_all(self.args.seed)

        # Make cuDNN deterministic (slower but reproducible)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # -----------------------
        # Setup dtype
        # -----------------------
        self.dtype = getattr(torch, self.args.dtype)

        # -----------------------
        # Model
        # -----------------------
        self.model = TransformerLM(
            vocab_size = self.args.vocab_size,
            context_length = self.args.context_length,
            num_layers = self.args.num_layers,
            d_model = self.args.d_model,
            num_heads = self.args.num_heads,
            d_ff = self.args.d_ff,
            rope_theta = self.args.rope_theta,
            device = self.args.device,
            dtype = self.dtype,
        )

        # -----------------------
        # Optimizer
        # -----------------------
        self.optimizer = AdamW(
            self.model.parameters(),
            lr = self.args.lr_max,
            betas = (self.args.betas1, self.args.betas2),
            eps = self.args.eps,
            weight_decay = self.args.weight_decay,
        )

        # -------------------------------
        # Resume checkpoint if available
        # -------------------------------
        self.start_step = 0
        if self.args.resume:
            self.start_step = load_checkpoint(
                src = self.args.checkpoint_path,
                model = self.model,
                optimizer = self.optimizer,
            )

            print(f"[Resume] Loaded checkpoint from {self.args.checkpoint_path} at step = {self.start_step}")


    @torch.inference_mode()
    def evaluate(
        self,
    ) -> tuple[float, float]:
        """
        Evaluate average loss and perplexity on validation set.

        Uses self.args:
            - model: TransformerLM.
            - valid_tokens: Path to validation .npy file.
            - batch_size: Batch size.
            - context_length: Context length.
            - device: Device string.
            - eval_batches: Number of validation batches.

        Returns:
            - val_loss: Mean validation loss.
            - val_ppl: Mean validation perplexity.
        """
        was_training = self.model.training
        self.model.eval()
        losses = []

        for _ in range(self.args.eval_batches):
            x, y = data_loader(
                x = self.args.valid_tokens,
                batch_size = self.args.batch_size,
                context_length = self.args.context_length,
                device = self.args.device,
            )
            logits = self.model(x)
            loss = cross_entropy_loss(logits, y)
            losses.append(loss.item())

        val_loss = float(np.mean(losses))
        val_ppl = float(np.exp(min(val_loss, 20))) # cap loss to prevent overflow in exp
        
        if was_training:
            self.model.train()
        
        return val_loss, val_ppl


    def train(self):
        # -----------------------
        # Initialize early stopping
        # -----------------------
        best_val_loss = float("inf")
        best_step = -1
        bad_evals = 0

        # evaluate once at start (especially if resuming)
        val_loss, val_ppl = self.evaluate()

        best_val_loss = val_loss
        best_step = self.start_step

        if self.args.resume:
            print(f"[Resume] initial val_loss = {val_loss:.4f} ppl = {val_ppl:.2f}")
        else:
            print(f"[Start] initial val_loss = {val_loss:.4f} ppl = {val_ppl:.2f}")

            save_checkpoint(
                self.model,
                self.optimizer,
                self.start_step,
                self.args.checkpoint_path,
                config=vars(self.args),
            )
            print(f"[checkpoint] saved initial checkpoint to {self.args.checkpoint_path}")

        # -----------------------
        # Training loop
        # -----------------------
        self.model.train()
        t0 = time.time()

        for step in range(self.start_step + 1, self.args.max_steps + 1):
            # scheduler updates optimizer lr
            lr_t = cosine_annealing_scheduler(
                t = step,
                alpha_max = self.args.lr_max,
                alpha_min = self.args.lr_min,
                T_w = self.args.warmup_steps,
                T_c = self.args.cosine_steps,
            )
            self.optimizer.set_lr(lr=lr_t)

            # sample batch
            x, y = data_loader(
                x = self.args.train_tokens,
                batch_size = self.args.batch_size,
                context_length = self.args.context_length,
                device = self.args.device,
            )

            # forward + loss
            logits = self.model(x)
            loss = cross_entropy_loss(logits, y)

            # backward
            self.optimizer.zero_grad()
            loss.backward()

            # gradient clipping
            grad_norm = gradient_clipping(
                params=self.model.parameters(),
                max_l2_norm=self.args.grad_clip,
            )

            # step
            self.optimizer.step()

            # logging
            if step % self.args.log_every == 0:
                dt = time.time() - t0
                ppl = float(torch.exp(torch.clamp(loss.detach().float(), max=20)).cpu().item())
                print(
                    f"[step {step:6d}] "
                    f"loss = {loss.item():.4f} ppl = {ppl:.2f} "
                    f"lr = {lr_t:.6e} grad_norm = {grad_norm.item():.4f} "
                    f"time = {dt:.2f} sec"
                )

            # -----------------------------------------------
            # Validation + early stopping + best checkpoint
            # -----------------------------------------------
            if step % self.args.eval_every == 0:
                val_loss, val_ppl = self.evaluate()
                print(f"[val] loss = {val_loss:.4f} ppl = {val_ppl:.2f}")

                improved = val_loss < (best_val_loss - self.args.early_stop_min_delta)

                if improved:
                    best_val_loss = val_loss
                    best_step = step
                    bad_evals = 0
                    print(f"[best] new best val_loss = {best_val_loss:.4f} at step = {best_step}")

                    save_checkpoint(
                        self.model,
                        self.optimizer,
                        step,
                        self.args.checkpoint_path,
                        config=vars(self.args),
                    )
                    print(f"[checkpoint] saved BEST checkpoint to {self.args.checkpoint_path}")

                else:
                    bad_evals += 1
                    print(
                        f"[early_stop] no improvement ({bad_evals}/{self.args.early_stop_patience}) "
                        f"best_val_loss = {best_val_loss:.4f} at step = {best_step}"
                    )

                    if bad_evals >= self.args.early_stop_patience:
                        print(
                            f"[early_stop] stopping training: "
                            f"best_val_loss = {best_val_loss:.4f} at step = {best_step}"
                        )
                        break

        print("Training finished.")


def main():
    trainer = Trainer()
    trainer.train()


if __name__ == "__main__":
    main()