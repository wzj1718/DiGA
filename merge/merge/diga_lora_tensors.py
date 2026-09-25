#!/usr/bin/env python3
"""Run the diga-tensors merge pipeline on LoRA adapters.

The LoRA adapters are first materialized as full expert models with
PeftModel.merge_and_unload(), then those experts are passed to diga-tensors.py.
The merge algorithm is fixed; the remaining options control inputs, expert
materialization, device placement and output.
"""

import argparse
import gc
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


DEFAULT_ADAPTER_PATHS = [
    "models/OrthoMerge_LoRA_Model/llama3-1_8b_finetune_magicoder",
    "models/OrthoMerge_LoRA_Model/llama3-1_8b_finetune_numinamath",
    "models/OrthoMerge_LoRA_Model/llama3-1_8b_finetune_commonsense",
    "models/OrthoMerge_LoRA_Model/llama3-1_8b_finetune_socialiqa",
    "models/OrthoMerge_LoRA_Model/llama3-1_8b_finetune_scienceqa",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", default="models/Llama-3.1-8B")
    parser.add_argument("--adapter_paths", nargs="+", default=DEFAULT_ADAPTER_PATHS)
    parser.add_argument(
        "--expert_names",
        nargs="+",
        default=None,
        help="Names used for temporary LoRA expert folders. Defaults are derived from adapter folders.",
    )
    parser.add_argument(
        "--output_path",
        default=str(PROJECT_DIR / "outputs" / "models" / "Llama-3.1-8B_diga_LoRA_tensors_ties"),
    )
    parser.add_argument(
        "--diga_script",
        default=str(PROJECT_DIR / "merge" / "diga-tensors.py"),
        help="diga tensor merge script to run after materializing experts.",
    )
    parser.add_argument(
        "--torch_dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Dtype used while materializing LoRA experts.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_shard_size", default="5GB")
    parser.add_argument("--python_bin", default=sys.executable)

    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def derive_expert_names(adapter_paths: List[str], expert_names: Optional[List[str]]) -> List[str]:
    if expert_names is not None:
        if len(expert_names) != len(adapter_paths):
            raise ValueError(
                f"--expert_names length {len(expert_names)} != "
                f"--adapter_paths length {len(adapter_paths)}"
            )
        return expert_names

    names = []
    for path in adapter_paths:
        name = Path(path.rstrip("/")).name
        prefix = "llama3-1_8b_finetune_"
        names.append(name[len(prefix):] if name.startswith(prefix) else name)
    return names


def materialize_lora_experts(args, expert_names: List[str], expert_root: Path) -> List[Path]:
    base_model = Path(args.base_model)

    dtype = dtype_from_name(args.torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        local_files_only=True,
        trust_remote_code=True,
    )

    expert_dirs = []
    for idx, (name, adapter_path) in enumerate(zip(expert_names, args.adapter_paths), 1):
        adapter_path = Path(adapter_path)
        out_dir = expert_root / name
        expert_dirs.append(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[LoRA expert {idx}/{len(args.adapter_paths)}] {adapter_path} -> {out_dir}")
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            device_map=None,
            local_files_only=True,
            trust_remote_code=True,
        )
        base.to(args.device)

        peft_model = PeftModel.from_pretrained(
            base,
            adapter_path,
            local_files_only=True,
        )
        merged_model = peft_model.merge_and_unload()
        merged_model.save_pretrained(
            out_dir,
            safe_serialization=True,
            max_shard_size=args.max_shard_size,
        )
        tokenizer.save_pretrained(out_dir)

        del merged_model, peft_model, base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return expert_dirs


def run_diga_tensors(args, expert_dirs: List[Path]) -> None:
    cmd = [
        args.python_bin, str(Path(args.diga_script)),
        "--base_model_path", str(Path(args.base_model)),
        "--expert_model_paths", *[str(path) for path in expert_dirs],
        "--output_path", str(Path(args.output_path)),
        "--device", args.device,
        "--seed", str(args.seed),
    ]
    print("[Run]", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(PROJECT_DIR))


def main() -> None:
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    args = parse_args()
    expert_names = derive_expert_names(args.adapter_paths, args.expert_names)

    print("=" * 80)
    print("diga LoRA tensors")
    print(f"Base model       : {args.base_model}")
    print(f"Output path      : {args.output_path}")
    print(f"diga script      : {args.diga_script}")
    print("=" * 80)

    with tempfile.TemporaryDirectory(prefix="difa_lora_experts_") as temp_dir:
        expert_dirs = materialize_lora_experts(args, expert_names, Path(temp_dir))
        run_diga_tensors(args, expert_dirs)


if __name__ == "__main__":
    main()
