#!/bin/zsh
# We implement Gemni as checkfreq style with Ramdisk for checkpointing
# Set environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export NCCL_IB_DISABLE=1

# Training parameters
DATASET=cifar100
MODEL=resnet50
EPOCHS=10
BATCH_SIZE=64
COMPRESSOR=topk
COMPRESSOR_RATIO=0.01
SAVE_DIR=/data/checkfreq

# Distributed training with DeepSpeed
deepspeed --hostfil=hostfile ./torch/gemini.py \
  --dataset $DATASET \
  --model $MODEL \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --compressor $COMPRESSOR \
  --compressor_ratio $COMPRESSOR_RATIO \
  --save-dir $SAVE_DIR 
