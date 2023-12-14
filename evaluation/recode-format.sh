#!/bin/bash

# Base command
BASE_CMD="accelerate launch main.py \
--tasks perturbed-humaneval-format-num_seeds_5 \
--use_auth_token \
--do_sample False \
--batch_size 1 \
--allow_code_execution \
--save_generations \
--trust_remote_code \
--n_samples 1 \
--max_length_generation 1024"

# File path
GENERATIONS_PATH="./generations-humaneval-format-num_seeds_5-starcoderbase"
EVAL_PATH="./evaluation-humaneval-format-num_seeds_5-starcoderbase"

PEFT_METHODS=("lora" "ptuning" "ia3" "adapterp" "adapterh" "parallel")
MODELS=("1b" "3b" "7b" "15b")

for MODEL in "${MODELS[@]}"; do
    for METHOD in "${PEFT_METHODS[@]}"; do
        GEN_FILE="${GENERATIONS_PATH}-${MODEL}-$METHOD.json"
        
        # Check if the generations file already exists
        if [[ -f $GEN_FILE ]]; then
            echo "Generations file for $MODEL-$METHOD already exists. Skipping..."
            continue
        fi

        # For 15b model, the name is "starcoderbase" not "starcoderbase-$MODEL"
        if [ "$MODEL" == "15b" ]; then
            MODEL_NAME="starcoderbase"
        else
            MODEL_NAME="starcoderbase-$MODEL"
        fi

        CMD="$BASE_CMD \
        --model bigcode/$MODEL_NAME \
        --peft_model bigcode/${MODEL_NAME/starcoderbase/astraios}-$METHOD \
        --save_generations_path $GEN_FILE \
        --metric_output_path ${EVAL_PATH}-${MODEL}-$METHOD.json"

        # Execute the command
        sh -c "$CMD"
        echo "-----------------------------------------"
    done

    GEN_FILE="${GENERATIONS_PATH}-${MODEL}-fft.json"
    
    # Check for "fft" method
    if [[ -f $GEN_FILE ]]; then
        echo "Generations file for $MODEL-fft already exists. Skipping..."
        continue
    fi

    # For 15b model, the name is "astraios" not "astraios-$MODEL"
    if [ "$MODEL" == "15b" ]; then
        FFT_MODEL_NAME="astraios-fft"
    else
        FFT_MODEL_NAME="astraios-$MODEL-fft"
    fi

    CMD="$BASE_CMD \
    --model bigcode/$FFT_MODEL_NAME \
    --save_generations_path $GEN_FILE \
    --metric_output_path ${EVAL_PATH}-${MODEL}-fft.json"

    # Execute the command for "fft" method
    sh -c "$CMD"
    echo "-----------------------------------------"
done
