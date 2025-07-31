#!/bin/zsh

# Set environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export NCCL_IB_DISABLE=1

# Training parameters
DATASET=wikitext-2
MODEL=gpt2-large
EPOCHS=10
BATCH_SIZE=2
COMPRESSOR=topk
COMPRESSOR_RATIO=0.01
FREQ=5
SAVE_BATCH_FREQ=1
SAVE_DIR=/data/ob
RESUME=0

# Distributed training with DeepSpeed
# deepspeed --hostfil=hostfile test.py
deepspeed --hostfil=hostfile storage_nlp.py \
  --dataset $DATASET \
  --model $MODEL \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --compressor $COMPRESSOR \
  --compressor_ratio $COMPRESSOR_RATIO \
  --diff \
  --freq $FREQ \
  --save-batch-freq $SAVE_BATCH_FREQ \
  --save-dir $SAVE_DIR \
  --resume $RESUME
