#!/bin/bash
LR=2e-4
WD=0
SPARSITY=0.95
PREPROCESS_NUM_SAMPLES=128
NUM_PARTITION=8
BLOCK_SIZE=64
OUTPUT_PATH=.cache
OUTPUT_DIR="$OUTPUT_PATH/sylva-llama-7b"

CMD="torchrun --nproc-per-node 1 --node-rank 0 main.py \
    --data_path timdettmers/openassistant-guanaco \
    --model_name_or_path huggyllama/llama-7b \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --max_seq_len 512 \
    --learning_rate $LR \
    --weight_decay $WD \
    --num_train_epochs 3 \
    --gradient_accumulation_steps 16 \
    --scheduler linear \
    --num_warmup_steps 100 \
    --seed 1234 \
    --preprocess_num_samples $PREPROCESS_NUM_SAMPLES \
    --num_partition $NUM_PARTITION \
    --sparsity $SPARSITY \
    --scope "self_attn" \
    --output_dir $OUTPUT_DIR \
    --log_interval 100 \
    --max_steps 13846
    --eval_interval 13846 \
    --dtype bf16 \
    --block_size $BLOCK_SIZE \
    --target-score 0.38"
echo $CMD
eval $CMD

