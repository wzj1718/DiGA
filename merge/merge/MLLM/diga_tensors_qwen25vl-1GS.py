#!/usr/bin/env python3
"""diga tensor merging for full Qwen2.5-VL checkpoints.

Target matrices use the fixed one-pass T/R operator; non-target tensors use
base + 0.2 * sum(expert - base). Checkpoints are streamed and saved in shards.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor
from tqdm import tqdm

PARENT_MERGE_DIR = Path(__file__).resolve().parents[1]
if str(PARENT_MERGE_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_MERGE_DIR))

from diga_fixed import (
    is_target_matrix, merge_matrix as diga_merge_matrix,
    merge_non_target_ta, seed_torch,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_BASE_MODEL = Path("models/Qwen2.5-VL-7B-Instruct")
DEFAULT_EXPERT_MODELS = [
    Path("models/SenseNova-SI-1.1-Qwen2.5-VL-7B"),
    Path("models/olmOCR-2-7B-1025"),
    Path("models/HuatuoGPT-Vision-7B-Qwen2.5VL"),
]

COPY_FILE_NAMES = {
    "added_tokens.json",
    "chat_template.jinja",
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
}


def parse_size(size: str) -> int:
    units = {
        "b": 1,
        "kb": 1000,
        "mb": 1000**2,
        "gb": 1000**3,
        "kib": 1024,
        "mib": 1024**2,
        "gib": 1024**3,
    }
    raw = size.strip().lower()
    for suffix in sorted(units, key=len, reverse=True):
        if raw.endswith(suffix):
            return int(float(raw[: -len(suffix)].strip()) * units[suffix])
    return int(raw)


def load_index(model_dir: Path) -> Dict[str, str]:
    index_file = model_dir / "model.safetensors.index.json"
    if not index_file.exists():
        raise FileNotFoundError(f"Missing safetensors index: {index_file}")
    return json.loads(index_file.read_text())["weight_map"]


def load_config(model_dir: Path) -> dict:
    return json.loads((model_dir / "config.json").read_text())


def get_tensor(model_dir: Path, index: Dict[str, str], key: str) -> Tensor:
    with safe_open(model_dir / index[key], framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def copy_model_assets(base_model: Path, output_dir: Path) -> None:
    for item in base_model.iterdir():
        if item.name in COPY_FILE_NAMES and item.is_file():
            shutil.copy2(item, output_dir / item.name)


def flush_shard(
    shard_tensors: OrderedDict[str, Tensor],
    tmp_paths: List[Path],
    shard_keys: List[List[str]],
    output_dir: Path,
) -> None:
    if not shard_tensors:
        return
    tmp_path = output_dir / f"tmp-shard-{len(tmp_paths) + 1:05d}.safetensors"
    save_file(shard_tensors, tmp_path, metadata={"format": "pt"})
    tmp_paths.append(tmp_path)
    shard_keys.append(list(shard_tensors.keys()))
    shard_tensors.clear()
    gc.collect()


def save_index(output_dir: Path, tmp_paths: List[Path], shard_keys: List[List[str]], total_size: int) -> None:
    weight_map = {}
    total = len(tmp_paths)
    for idx, tmp_path in enumerate(tmp_paths, 1):
        final_name = f"model-{idx:05d}-of-{total:05d}.safetensors"
        final_path = output_dir / final_name
        tmp_path.rename(final_path)
        for key in shard_keys[idx - 1]:
            weight_map[key] = final_name
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (output_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")


def validate_compatibility(base_model: Path, expert_models: Sequence[Path], keys: Sequence[str]) -> None:
    base_cfg = load_config(base_model)
    base_index = load_index(base_model)
    required_fields = ["model_type", "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "vocab_size"]
    for expert in expert_models:
        cfg = load_config(expert)
        for field in required_fields:
            if cfg.get(field) != base_cfg.get(field):
                raise ValueError(f"Config mismatch for {expert} field {field}: {cfg.get(field)!r} != {base_cfg.get(field)!r}")
        expert_index = load_index(expert)
        if set(expert_index) != set(base_index):
            raise ValueError(f"State-dict key mismatch for {expert}")
    print("[Check] Key sets and core configs are compatible.")
    for key in tqdm(keys, desc="[Check] Shapes/dtypes"):
        base_tensor = get_tensor(base_model, base_index, key)
        base_shape, base_dtype = tuple(base_tensor.shape), base_tensor.dtype
        del base_tensor
        for expert in expert_models:
            tensor = get_tensor(expert, load_index(expert), key)
            if tuple(tensor.shape) != base_shape:
                raise ValueError(f"Shape mismatch for {key} in {expert}")
            if tensor.dtype != base_dtype:
                raise ValueError(f"Dtype mismatch for {key} in {expert}")
            del tensor


def should_process_key(key: str, tensor: Tensor) -> bool:
    return torch.is_floating_point(tensor) and key.startswith(("visual.", "model.", "lm_head."))


def merge_models(args: argparse.Namespace) -> None:
    seed_torch(args.seed)
    base_model = Path(args.base_model_path)
    expert_models = [Path(path) for path in args.expert_model_paths]
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_index = load_index(base_model)
    expert_indexes = [load_index(path) for path in expert_models]
    keys = list(base_index.keys())
    if args.validate:
        validate_compatibility(base_model, expert_models, keys)

    print(f"[Merge] Base: {base_model}")
    print(f"[Merge] Experts (in order): {expert_models}")
    print(f"[Merge] Output: {output_dir}")

    copy_model_assets(base_model, output_dir)
    max_shard_size = parse_size(args.max_shard_size)
    shard_tensors: OrderedDict[str, Tensor] = OrderedDict()
    shard_size = 0
    total_size = 0
    tmp_paths: List[Path] = []
    shard_keys: List[List[str]] = []
    diga_count = 0
    non_target_count = 0
    copied_count = 0

    for key in tqdm(keys, desc="[diga-MLLM] Tensors"):
        base_tensor = get_tensor(base_model, base_index, key)
        process_this = should_process_key(key, base_tensor)

        if process_this:
            expert_tensors = [get_tensor(path, index, key) for path, index in zip(expert_models, expert_indexes)]
            if is_target_matrix(key, base_tensor, multimodal=True):
                tensor = diga_merge_matrix(base_tensor, expert_tensors, args.device)
                diga_count += 1
            else:
                tensor = merge_non_target_ta(base_tensor, expert_tensors, args.device)
                non_target_count += 1
            del expert_tensors
        else:
            tensor = base_tensor.cpu()
            copied_count += 1

        tensor_size = tensor.numel() * tensor.element_size()
        if shard_tensors and shard_size + tensor_size > max_shard_size:
            flush_shard(shard_tensors, tmp_paths, shard_keys, output_dir)
            shard_size = 0
        shard_tensors[key] = tensor.contiguous()
        shard_size += tensor_size
        total_size += tensor_size
        del base_tensor, tensor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    flush_shard(shard_tensors, tmp_paths, shard_keys, output_dir)
    save_index(output_dir, tmp_paths, shard_keys, total_size)
    print(f"[Done] diga tensors: {diga_count}, non-target residual tensors: {non_target_count}, copied tensors: {copied_count}")
    print(f"[Done] wrote {len(tmp_paths)} shards to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model_path", default=str(DEFAULT_BASE_MODEL))
    parser.add_argument("--expert_model_paths", nargs="+", default=[str(path) for path in DEFAULT_EXPERT_MODELS])
    parser.add_argument("--output_path", default=str(PROJECT_DIR / "outputs" / "models" / "Qwen2.5-VL_diga_MLLM_tensors"))
    parser.add_argument("--max_shard_size", default="5GB")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate", action="store_true")
    return parser


def main() -> None:
    merge_models(build_parser().parse_args())


if __name__ == "__main__":
    main()
