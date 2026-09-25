#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

# The project constructs an OpenAI client at import time. A dummy key is enough
# for non-OpenAI safety tasks; set a real key if you use GPT/OpenAI judges.
export OPENAI_API_KEY=${OPENAI_API_KEY:-dummy}

SAFETY_EVAL_DIR="safety-eval-fork"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESULT_ROOT="results/safety"

MODEL_INPUT_TEMPLATE=${MODEL_INPUT_TEMPLATE:-llama3}
SAFETY_TASKS=${SAFETY_TASKS:-wildguardtest,harmbench,xstest,do_anything_now}
BATCH_SIZE=${BATCH_SIZE:-8}

models=( )

cd "$SAFETY_EVAL_DIR"

for model_path in "${models[@]}"; do
  if [ ! -d "$model_path" ]; then
    echo "Skip missing model dir: $model_path" >&2
    continue
  fi

  model_name=$(basename "$model_path")
  output_path="$RESULT_ROOT/${model_name}/safety"
  mkdir -p "$output_path"

  echo "============================================================"
  echo "MODEL: $model_path"
  echo "TASKS: $SAFETY_TASKS"
  echo "OUT  : $output_path"
  echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
  echo "============================================================"

  "$PYTHON_BIN" evaluation/eval.py generators \
    --model_name_or_path "$model_path" \
    --model_input_template_path_or_name "$MODEL_INPUT_TEMPLATE" \
    --tasks "$SAFETY_TASKS" \
    --report_output_path "$output_path/safety_eval.json" \
    --save_individual_results_path "$output_path/safety_generation.json" \
    --batch_size "$BATCH_SIZE" \
    "$@"
done
