#!/bin/bash
LANGS=("python" "java")

for lang in "${LANGS[@]}"; do
    # Base command
    BASE_CMD="accelerate launch main.py \
    --tasks humanevalexplainsynthesize-$lang \
    --do_sample True \
    --temperature 0.2 \
    --n_samples 1 \
    --batch_size 1 \
    --allow_code_execution \
    --save_generations \
    --trust_remote_code \
    --prompt octocoder \
    --use_auth_token \
    --max_length_generation 2048"

    # File path
    GENERATIONS_PATH="./generations-humanevalexplainsynthesize-$lang-starcoderbase"
    EVAL_PATH="./evaluation-humanevalexplainsynthesize-$lang-starcoderbase"

    PEFT_METHODS=()
    MODELS=("1b" "3b" "7b" "15b")

    for MODEL in "${MODELS[@]}"; do
        for METHOD in "${PEFT_METHODS[@]}"; do
            GEN_FILE="${GENERATIONS_PATH}-${MODEL}-$METHOD.json"
            
            # Check if the generations file already exists
            if [[ -f $GEN_FILE ]]; then
                echo "Generations file for $MODEL-$METHOD already exists. Skipping..."
                continue
            fi
            if [[ ! -f generations-humanevalexplaindescribe-$lang-starcoderbase-${MODEL}-$METHOD.json ]]; then
                # echo "Generations file for $MODEL-$METHOD does not exist. Skipping..."
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
            --peft_model starpeft/$MODEL_NAME-$METHOD \
            --save_generations_path $GEN_FILE \
            --load_data_path generations-humanevalexplaindescribe-$lang-starcoderbase-${MODEL}-$METHOD.json \
            --metric_output_path ${EVAL_PATH}-${MODEL}-$METHOD.json"

            # Execute the command
            sh -c "$CMD"
            echo "-----------------------------------------"
        done

        GEN_FILE="${GENERATIONS_PATH}-${MODEL}-fft.json"
        
        if [[ ! -f generations-humanevalexplaindescribe-$lang-starcoderbase-${MODEL}-fft.json ]]; then
                # echo "Generations file for $MODEL-fft does not exist. Skipping..."
                continue
        fi
        # For 15b model, the name is "starcoderbase" not "starcoderbase-$MODEL"
        if [ "$MODEL" == "15b" ]; then
            FFT_MODEL_NAME="starcoderbase-fft"
        else
            FFT_MODEL_NAME="starcoderbase-$MODEL-fft"
        fi

        CMD="$BASE_CMD \
        --model starpeft/$FFT_MODEL_NAME \
        --save_generations_path $GEN_FILE \
        --load_data_path generations-humanevalexplaindescribe-$lang-starcoderbase-${MODEL}-fft.json \
        --metric_output_path ${EVAL_PATH}-${MODEL}-fft.json"

        # Execute the command for "fft" method
        echo ${MODEL}-fft
        sh -c "$CMD"
        echo "-----------------------------------------"
    done
done