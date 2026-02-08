#!/bin/bash
set -e

# ==========================================
# GPT2-medium
# ==========================================

export CUDA_VISIBLE_DEVICES=3

TRAIN_TOKENS="./cs336_basics/BPE_Tokenizer/tests/TinyStoriesV2-GPT4-train.npy"
VALID_TOKENS="./cs336_basics/BPE_Tokenizer/tests/TinyStoriesV2-GPT4-valid.npy"
CHECKPOINT_PATH="./scripts/checkpoints/train_on_gpu_$(date +%Y%m%d_%H%M%S).pt"
LOG_PATH="./scripts/logs/train_on_gpu_$(date +%Y%m%d_%H%M%S).log"

mkdir -p ./scripts/checkpoints
mkdir -p ./scripts/logs

PYTHONUNBUFFERED=1 python ./cs336_basics/Trainer/training_loop.py \
  --train_tokens "$TRAIN_TOKENS" \
  --val_tokens "$VALID_TOKENS" \
  --device cuda \
  --dtype bfloat16 \
  --batch_size 8 \
  --context_length 1024 \
  --max_iters 20000 \
  --vocab_size 10000 \
  --num_layers 24 \
  --d_model 1024 \
  --num_heads 16 \
  --d_ff 4096 \
  --rope_theta 10000.0 \
  --lr_max 3e-4 \
  --lr_min 3e-5 \
  --warmup_iters 200 \
  --cosine_iters 20000 \
  --weight_decay 0.1 \
  --grad_clip 1.0 \
  --log_every 10 \
  --eval_every 100 \
  --eval_batches 50 \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --early_stop_patience 10 \
  --early_stop_min_delta 0.001 \
  --seed 51 \
  2>&1 | tee "$LOG_PATH"