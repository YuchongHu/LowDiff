#!/bin/bash

export MASTER_ADDR=192.168.3.18
export MASTER_PORT=29500
export NCCL_IB_DISABLE=1

deepspeed --hostfil=hostfile ./torch/pipeline.py -p 2 --steps=1000