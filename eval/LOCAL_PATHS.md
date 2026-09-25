# Local Evaluation Paths

The release does not contain machine-specific model or dataset locations.
Populate `models` in the scripts before running them. Start the scripts from
this `eval` directory, as their harness paths are relative to it.

Activate the appropriate environment first. `PYTHON_BIN` and `ACCELERATE_BIN`
can override the executables; by default they are resolved from `PATH`.
Safety evaluation still requires its own dependencies.

## Datasets and Models

Set `DIFA_DATA_ROOT` to an absolute directory containing your local datasets.
If unset, it defaults to `./data` relative to the process working directory
(normally the selected harness directory after the script's `cd`).
For example, the GSM8K dataset is read from `DIFA_DATA_ROOT/gsm8k`, and the
MMMU dataset from `DIFA_DATA_ROOT/MLLM/MMMU`.

The two YAML-based harnesses support a path tag:

```yaml
dataset_path: !env [DIFA_DATA_ROOT, ./data, gsm8k]
```

The first item is the environment variable, the second its fallback value,
and any remaining items are appended as path components. This is supported
by the included harness loaders, including task metadata safe-loading.
These configs require this bundled loader support, not an unmodified upstream
harness. Dataset names, splits, prompts, and evaluation settings are unchanged.

Additional overrides:

| Variable | Purpose / Default |
| --- | --- |
| `SCIENCEQA_DATASET_PATH` | ScienceQA path; otherwise `DIFA_DATA_ROOT/ScienceQA` |
| `HUMANEVALPLUS_DATASET_PATH` | BigCode HumanEval+ path; otherwise `DIFA_DATA_ROOT/evalplus-humanevalplus` |
| `MBPPPLUS_DATASET_PATH` | BigCode MBPP+ path; otherwise `DIFA_DATA_ROOT/mbppplus` |
| `SOCIALIQA_DATASET_PATH` | Dataset loader directory; defaults to `./lm_eval/tasks/siqa/social_i_qa` |
| `SOCIALIQA_DATA_DIR` | Raw SocialIQA data; defaults to the bundled directory beside its loader |
| `HF_DATASETS_CACHE` | PathVQA dataset cache; defaults to `./.hf_datasets_cache` |
| `CODE_EVAL_METRIC_PATH` | Local `evaluate` code metric; defaults to public `code_eval` |
| `WILDGUARD_MODEL_PATH` | Local WildGuard model; defaults to `./models/allenai/wildguard`, with the existing `allenai/wildguard` fallback |
| `WILDGUARDMIX_TEST_PATH` | Local parquet; defaults to `DIFA_DATA_ROOT/wildguardmix/test/wildguard_test.parquet`, with the existing public dataset fallback |
| `CHARTQA_DATA_DIR` | ChartQA preparation input; defaults to `./data/ChartQA/ChartQA Dataset` |

The scripts retain their offline settings. Ensure all required datasets,
models, and metrics are available locally or cached before running. In
particular, point `CODE_EVAL_METRIC_PATH` at your offline metric copy if needed.

The Qwen example uses `MODEL_PATH`, `MEDICAL_EVAL_ENV`, `MEDICAL_EVAL_DIR`, and
`MEDICAL_EVAL_DATA` for its optional local setup. LiveBench preparation tools
accept `LIVE_BENCH_WEBSITES`, `LIVE_BENCH_INPUT_DIR`, and `LIVE_BENCH_OUTPUT_DIR`.
The Slurm example accepts `BASH_INIT_FILE`, `EVAL_CONDA_ENV`, and
`BIGCODE_EVAL_DIR`; configure scheduler resources for your own cluster.

## Release Hygiene

Do not package `__pycache__`, `.pyc`, `.cache`, `.egg-info`, or notebook
checkpoints. Clear notebook outputs and execution metadata before publishing.
The `.gitignore` excludes generated caches, but a raw directory archive must
also exclude them. Keep upstream licenses, citations, and author attribution.
