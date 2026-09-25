#!/bin/bash

set -e

export CUDA_VISIBLE_DEVICES=3
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HF_ALLOW_CODE_EVAL=1

cd lm-evaluation-harness

models=( )


tasks=(
    gsm8k_cot


)

for model_path in "${models[@]}"; do
    model_name=$(basename ${model_path})

    for task in "${tasks[@]}"; do
        output_path=results/${model_name}/${task}
        mkdir -p ${output_path}

        echo "============================================================"
        echo "MODEL: ${model_path}"
        echo "TASK : ${task}"
        echo "OUT  : ${output_path}"
        echo "============================================================"

        "${PYTHON_BIN:-python}" -m lm_eval \
            --model hf \
            --model_args pretrained=${model_path},dtype=bfloat16,trust_remote_code=True \
            --tasks ${task} \
            --device cuda:0 \
            --batch_size 16 \
            --output_path ${output_path} \
            --confirm_run_unsafe_code
    done
done
            
