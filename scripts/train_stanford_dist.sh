#!/bin/bash

set -x
# Exit script when a command returns nonzero state
set -e

set -o pipefail

export PYTHONUNBUFFERED=True

export BATCH_SIZE=${BATCH_SIZE:-6}

export TIME=$(date +"%Y-%m-%d_%H-%M-%S")

export LOG_DIR=./outputs/StanfordArea5Dataset/$1/$TIME

# Save the experiment detail and dir to the common log file
mkdir -p $LOG_DIR

LOG="$LOG_DIR/$TIME.txt"

python -m main_distributed \
       --dataset StanfordArea5Dataset \
       --batch_size $BATCH_SIZE \
       --scheduler PolyLR \
       --model Res16UNet34 \
       --conv1_kernel_size 3 \
       --log_dir $LOG_DIR \
       --lr 1e-1 \
       --max_iter 20000 \
       --data_aug_color_trans_ratio 0.05 \
       --data_aug_color_jitter_std 0.005 \
       --train_phase train \
       --dist-url 'tcp://127.0.0.1:10001' \
       --dist-backend 'nccl' \
       --multiprocessing-distributed \
       --world-size 1 \
       --rank 0 \
       $2 2>&1 | tee -a
