#!/bin/zsh

# Set environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export NCCL_IB_DISABLE=1

# Training parameters
DATASET=imagenet
MODEL=resnet101
EPOCHS=10
BATCH_SIZE=64
COMPRESSOR=topk
COMPRESSOR_RATIO=0.01
FREQ=50
SAVE_BATCH_FREQ=1
SAVE_DIR=/data/lowdiff
RESUME=0

# Distributed training with DeepSpeed
deepspeed --hostfil=hostfile ./torch/cv.py \
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
