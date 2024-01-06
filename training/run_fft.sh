#!/bin/bash
models=("-1b" "-3b" "-7b" "")

for model in "${models[@]}"
do
    python finetune.py \
        --model_path="bigcode/starcoderbase$model" \
        --dataset_name="bigcode/guanaco-commits" \
        --input_column_name prompt \
        --output_column_name completion \
        --seq_length 2048 \
        --max_steps 200 \
        --batch_size 1 \
        --gradient_accumulation_steps 32 \
        --learning_rate 1e-4 \
        --lr_scheduler_type="cosine" \
        --num_warmup_steps 12 \
        --log_freq 5 \
        --eval_freq 10 \
        --save_freq 10 \
        --weight_decay 0.05 \
        --output_dir="./astraios$model-fft"
done
