#!/bin/bash
set -e

# ==========================================
# Small CPU training test (fast sanity run)
# ==========================================

TRAIN_TOKENS="./cs336_basics/BPE_Tokenizer/tests/TinyStoriesV2-GPT4-train.npy"
VALID_TOKENS="./cs336_basics/BPE_Tokenizer/tests/TinyStoriesV2-GPT4-valid.npy"
CHECKPOINT_PATH="./scripts/checkpoints/train_on_cpu_$(date +%Y%m%d_%H%M%S).pt"
LOG_PATH="./scripts/logs/train_on_cpu_$(date +%Y%m%d_%H%M%S).log"

mkdir -p ./scripts/checkpoints
mkdir -p ./scripts/logs

python -u ./cs336_basics/Trainer/training_loop.py \
  --train_tokens "$TRAIN_TOKENS" \
  --val_tokens "$VALID_TOKENS" \
  --device cpu \
  --dtype float32 \
  --batch_size 16 \
  --context_length 128 \
  --max_iters 1000 \
  --vocab_size 10000 \
  --num_layers 4 \
  --d_model 128 \
  --num_heads 4 \
  --d_ff 320 \
  --rope_theta 10000.0 \
  --lr_max 1e-3 \
  --lr_min 1e-4 \
  --warmup_iters 100 \
  --cosine_iters 1000 \
  --weight_decay 0.01 \
  --grad_clip 1.0 \
  --log_every 5 \
  --eval_every 25 \
  --eval_batches 20 \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --early_stop_patience 10 \
  --early_stop_min_delta 0.001 \
  --seed 51 \
  2>&1 | tee "$LOG_PATH"