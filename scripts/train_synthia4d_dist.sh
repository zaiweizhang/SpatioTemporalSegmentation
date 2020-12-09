#!/bin/bash

set -x
# Exit script when a command returns nonzero state
set -e

set -o pipefail

export PYTHONUNBUFFERED="True"

export BATCH_SIZE=${BATCH_SIZE:-7}
export DATASET=${DATASET:-SynthiaCVPR15cmVoxelizationDataset}
export MODEL=${MODEL:-Res16UNet34}

export TIME=$(date +"%Y-%m-%d_%H-%M-%S")

export LOG_DIR=./outputs/Synthia4D$1/$TIME

# Save the experiment detail and dir to the common log file
mkdir -p $LOG_DIR

LOG="$LOG_DIR/$TIME.txt"

python -m main_distributed \
    --log_dir $LOG_DIR \
    --dataset $DATASET \
    --model $MODEL \
    --conv1_kernel_size 3 \
    --lr 1e-1 \
    --batch_size $BATCH_SIZE \
    --scheduler PolyLR \
    --max_iter 20000 \
    --train_limit_numpoints 1200000 \
    --train_phase train \
    --dist-url 'tcp://127.0.0.1:10001' \
    --dist-backend 'nccl' \
    --multiprocessing-distributed \
    --world-size 1 \
    --rank 0 \
    $2 2>&1 | tee -a "$LOG"
