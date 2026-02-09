#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=3

TRAIN_TOKENS="./cs336_basics/BPE_Tokenizer/tests/owt_train.npy"
VALID_TOKENS="./cs336_basics/BPE_Tokenizer/tests/owt_valid.npy"
CHECKPOINT_PATH="./scripts/checkpoints/train_on_gpu_OpenWebText_$(date +%Y%m%d_%H%M%S).pt"
LOG_PATH="./scripts/logs/train_on_gpu_OpenWebText_$(date +%Y%m%d_%H%M%S).log"

mkdir -p ./scripts/checkpoints
mkdir -p ./scripts/logs

PYTHONUNBUFFERED=1 python ./cs336_basics/Trainer/trainer.py \
  --train_tokens "$TRAIN_TOKENS" \
  --valid_tokens "$VALID_TOKENS" \
  --device cuda \
  --dtype bfloat16 \
  --batch_size 30 \
  --context_length 512 \
  --max_steps 50000 \
  --vocab_size 32000 \
  --num_layers 24 \
  --d_model 1024 \
  --num_heads 16 \
  --d_ff 2720 \
  --rope_theta 10000.0 \
  --lr_max 6e-4 \
  --lr_min 3e-5 \
  --warmup_steps 5000 \
  --cosine_steps 50000 \
  --weight_decay 0.01 \
  --grad_clip 1.0 \
  --log_every 20 \
  --eval_every 200 \
  --eval_batches 100 \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --early_stop_patience 20 \
  --early_stop_min_delta 0.0001 \
  --seed 51 \
  2>&1 | tee "$LOG_PATH"