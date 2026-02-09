#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0

TRAIN_TOKENS="./cs336_basics/BPE_Tokenizer/tests/TinyStoriesV2-GPT4-train.npy"
VALID_TOKENS="./cs336_basics/BPE_Tokenizer/tests/TinyStoriesV2-GPT4-valid.npy"
CHECKPOINT_PATH="./scripts/checkpoints/train_on_gpu_TinyStories_$(date +%Y%m%d_%H%M%S).pt"
LOG_PATH="./scripts/logs/train_on_gpu_TinyStories_$(date +%Y%m%d_%H%M%S).log"

mkdir -p ./scripts/checkpoints
mkdir -p ./scripts/logs

PYTHONUNBUFFERED=1 python ./cs336_basics/Trainer/trainer.py \
  --train_tokens "$TRAIN_TOKENS" \
  --valid_tokens "$VALID_TOKENS" \
  --device cuda \
  --dtype bfloat16 \
  --batch_size 32 \
  --context_length 256 \
  --max_steps 30000 \
  --vocab_size 10000 \
  --num_layers 20 \
  --d_model 1024 \
  --num_heads 16 \
  --d_ff 2688 \
  --rope_theta 10000.0 \
  --lr_max 5e-4 \
  --lr_min 3e-5 \
  --warmup_steps 1000 \
  --cosine_steps 30000 \
  --weight_decay 0.1 \
  --grad_clip 1.0 \
  --log_every 20 \
  --eval_every 200 \
  --eval_batches 100 \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --early_stop_patience 10 \
  --early_stop_min_delta 0.001 \
  --seed 51 \
  2>&1 | tee "$LOG_PATH"