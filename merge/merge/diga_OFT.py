#!/usr/bin/env python3
"""Merge OFT experts with a fixed base-frame skew/residual algorithm.

Each target update is split into T and R. Each branch uses source-norm-weighted
Gram--Schmidt directions and magnitude (1 + mean_cosine) * mean_source_norm.
The two branches are added directly to the base weight. Non-target floating
parameters retain the original base-plus-experts mean rule.
"""

import argparse
import gc
import os
import random
from collections import OrderedDict
from fnmatch import fnmatch
from typing import Dict, List, Sequence, Tuple

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


EPS = 1e-8
EMBED_KEYS = ("model.embed_tokens.weight", "lm_head.weight")
MATRIX_INCLUDE_PATTERNS = (
    "*.q_proj.weight", "*.k_proj.weight", "*.v_proj.weight", "*.o_proj.weight",
    "*.gate_proj.weight", "*.up_proj.weight", "*.down_proj.weight",
    "*.fc1.weight", "*.fc2.weight",
    "*.c_attn.weight", "*.c_proj.weight",
    "*.dense_h_to_4h.weight", "*.dense_4h_to_h.weight",
)
MATRIX_EXCLUDE_PATTERNS = (
    "*.embed_tokens.weight", "*.lm_head.weight",
    "*.layernorm*.weight", "*.norm.weight",
)
_TOKENIZER_CACHE: Dict[str, object] = {}


def seed_torch(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_tokenizer(model_path: str):
    if model_path not in _TOKENIZER_CACHE:
        _TOKENIZER_CACHE[model_path] = AutoTokenizer.from_pretrained(model_path)
    return _TOKENIZER_CACHE[model_path]


def remap_embedding_to_base_vocab(
    base_vocab: Dict[str, int],
    src_vocab: Dict[str, int],
    src_embed: Tensor,
) -> Tensor:
    vocab_size = len(base_vocab)
    embed_size = src_embed.shape[1]
    device = src_embed.device
    dtype = src_embed.dtype
    new_embed = torch.zeros((vocab_size, embed_size), dtype=dtype, device=device)
    for token, base_id in base_vocab.items():
        src_id = src_vocab.get(token)
        if src_id is not None and src_id < src_embed.shape[0]:
            new_embed[base_id] = src_embed[src_id]
    return new_embed


def is_target_matrix(name: str, tensor: Tensor) -> bool:
    if tensor.ndim != 2:
        return False
    if "layernorm" in name.lower() or "norm.weight" in name.lower():
        return False
    if any(fnmatch(name, pattern) for pattern in MATRIX_EXCLUDE_PATTERNS):
        return False
    return any(fnmatch(name, pattern) for pattern in MATRIX_INCLUDE_PATTERNS)


def normalize(vec: Tensor, eps: float = EPS) -> Tuple[Tensor, Tensor]:
    norm = torch.linalg.vector_norm(vec)
    if float(norm.item()) <= eps:
        return torch.zeros_like(vec), norm
    return vec / norm, norm


def decompose_matrix_tangent_delta(
    base_matrix: Tensor, delta_matrix: Tensor,
) -> Tuple[Tensor, Tensor]:
    rows, cols = base_matrix.shape
    if rows >= cols:
        basis, _ = torch.linalg.qr(base_matrix, mode="reduced")
        generator = basis.transpose(0, 1) @ delta_matrix
        skew = 0.5 * (generator - generator.transpose(0, 1))
        tangent = basis @ skew
    else:
        basis, _ = torch.linalg.qr(base_matrix.transpose(0, 1), mode="reduced")
        generator = delta_matrix @ basis
        skew = 0.5 * (generator - generator.transpose(0, 1))
        tangent = skew @ basis.transpose(0, 1)
    residual = delta_matrix - tangent
    return tangent, residual


def orthogonalize_directions(
    directions: Sequence[Tensor],
    eps: float = EPS,
) -> List[Tensor]:
    """One-pass modified Gram--Schmidt, retaining the original fallback."""
    ortho_dirs: List[Tensor] = []
    for direction in directions:
        work = direction.clone()
        for previous in ortho_dirs:
            work = work - torch.dot(work, previous) * previous
        work, norm = normalize(work, eps=eps)
        if float(norm.item()) <= eps:
            print("  [ortho] fallback - using original direction")
            fallback, fallback_norm = normalize(direction, eps=eps)
            work = fallback if float(fallback_norm.item()) > eps else torch.zeros_like(direction)
        ortho_dirs.append(work)
    return ortho_dirs


def merge_direction_family(
    deltas: Sequence[Tensor],
    norms: Tensor,
    eps: float = EPS,
) -> Tensor:
    """Construct the branch direction, then set its consensus-aware magnitude."""
    if not deltas:
        return torch.zeros(0)
    if torch.all(norms <= eps):
        return torch.zeros_like(deltas[0])

    unit_dirs = [
        delta / norm if float(norm.item()) > eps else torch.zeros_like(delta)
        for delta, norm in zip(deltas, norms)
    ]
    ortho_dirs = orthogonalize_directions(unit_dirs, eps=eps)

    # L2-normalized source weights give the same final unit direction as n_i.
    weights = norms / torch.linalg.vector_norm(norms)
    merged_dir = torch.zeros_like(deltas[0])
    for weight, direction in zip(weights, ortho_dirs):
        merged_dir = merged_dir + weight * direction
    merged_dir, merged_norm = normalize(merged_dir, eps=eps)
    if float(merged_norm.item()) <= eps:
        print("  [merge] direction ~0, returning zero delta")
        return torch.zeros_like(deltas[0])

    if len(unit_dirs) > 1:
        unit_matrix = torch.stack(unit_dirs, dim=0)
        similarities = unit_matrix @ unit_matrix.T
        mask = ~torch.eye(len(unit_dirs), dtype=torch.bool, device=similarities.device)
        mean_cosine = similarities[mask].mean().clamp(-1.0, 1.0)
    else:
        # Match the original single-source convention (full agreement).
        mean_cosine = norms.new_tensor(1.0)

    magnitude = float(((1.0 + mean_cosine) * norms.mean()).item())
    return merged_dir * magnitude


def diga_merge_matrix(
    base_tensor: Tensor,
    expert_tensors: Sequence[Tensor],
    device: str,
) -> Tensor:
    base_work = base_tensor.to(device=device, dtype=torch.float32)
    base_flat = base_work.reshape(-1)
    raw_deltas = []
    tangents = []
    residuals = []
    for expert_tensor in expert_tensors:
        delta = expert_tensor.to(device=device, dtype=torch.float32) - base_work
        tangent, residual = decompose_matrix_tangent_delta(base_work, delta)
        raw_deltas.append(delta.reshape(-1))
        tangents.append(tangent.reshape(-1))
        residuals.append(residual.reshape(-1))

    delta_norms = torch.stack([torch.linalg.vector_norm(delta) for delta in raw_deltas])
    if torch.all(delta_norms <= EPS):
        return base_tensor.clone()

    tangent_norms = torch.stack([torch.linalg.vector_norm(tangent) for tangent in tangents])
    if torch.all(tangent_norms <= EPS):
        # Preserve the original whole-update mean fallback for an empty T branch.
        print("  [diga] fallback to mean delta")
        mean_delta = torch.stack(raw_deltas, dim=0).mean(dim=0)
        return (base_flat + mean_delta).reshape_as(base_work).to(dtype=base_tensor.dtype).cpu()

    merged_tangent = merge_direction_family(tangents, tangent_norms)
    if float(torch.linalg.vector_norm(merged_tangent).item()) <= EPS:
        return base_tensor.clone()

    residual_norms = torch.stack([torch.linalg.vector_norm(residual) for residual in residuals])
    if torch.any(residual_norms > EPS):
        merged_residual = merge_direction_family(residuals, residual_norms)
    else:
        merged_residual = torch.stack(residuals, dim=0).mean(dim=0)

    merged = base_flat + (merged_tangent + merged_residual)
    return merged.reshape_as(base_work).to(dtype=base_tensor.dtype).cpu()


def mean_merge_tensor(base_tensor: Tensor, expert_tensors: Sequence[Tensor]) -> Tensor:
    stack = torch.stack(
        [base_tensor] + [tensor.to(dtype=base_tensor.dtype) for tensor in expert_tensors],
        dim=0,
    )
    return stack.mean(dim=0)


def merge_non_target_tensor(base_tensor: Tensor, expert_tensors: Sequence[Tensor]) -> Tensor:
    """Keep the original base_mean arithmetic for non-target parameters."""
    mean_tensor = mean_merge_tensor(base_tensor, expert_tensors)
    return (base_tensor + (mean_tensor - base_tensor)).to(dtype=base_tensor.dtype)


def load_expert_state_dict_from_oft(
    base_model_name: str,
    adapter_path: str,
    device: str = "cpu",
) -> Dict[str, Tensor]:
    """Convert an OFT adapter to an in-memory bf16 expert state dict."""

    print(f"  [OFT→Expert] Loading base + adapter: {adapter_path}")

    # Load base model in float32 to match OFT adapter dtype (float32).
    # bfloat16 would cause dtype mismatch in PEFT's OFT layer merge.
    base_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=base_model_name,
        torch_dtype=torch.float32,
        device_map=None,
    )

    peft_model = PeftModel.from_pretrained(base_model, adapter_path)
    merged_model = peft_model.merge_and_unload()
    merged_model.to(device)

    state_dict = {k: v.cpu().to(torch.bfloat16) for k, v in merged_model.state_dict().items()}

    del peft_model, merged_model, base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return state_dict


def merge_models(args: argparse.Namespace) -> None:
    seed_torch(args.seed)
    os.makedirs(args.output_path, exist_ok=True)

    print(f"[Load] Base model: {args.language_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.language_model_name,
        device_map="cpu",
        torch_dtype=torch.bfloat16,
    )
    base_state = base_model.state_dict()
    expert_states: List[Dict[str, Tensor]] = []
    for i, adapter_path in enumerate(args.adapter_paths, 1):
        print(f"\n[{i}/{len(args.adapter_paths)}] Converting OFT adapter to full model...")
        expert_states.append(load_expert_state_dict_from_oft(
            args.language_model_name, adapter_path, device=args.device,
        ))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    base_vocab = get_tokenizer(args.language_model_name).get_vocab()
    for i, expert_state in enumerate(expert_states):
        expert_vocab = get_tokenizer(args.adapter_paths[i]).get_vocab()
        for key in EMBED_KEYS:
            if key in expert_state and key in base_state:
                if expert_state[key].shape != base_state[key].shape:
                    print(f"  [Align] Remapping vocab for expert {i}, key={key}")
                    expert_state[key] = remap_embedding_to_base_vocab(
                        base_vocab, expert_vocab, expert_state[key],
                    )

    print("[Merge] Fixed T/R decomposition; source-norm-weighted GS; consensus magnitude")
    print("[Merge] Non-target floating tensors: base-plus-experts mean")
    merged_state = OrderedDict()
    for idx, (key, base_tensor) in enumerate(base_state.items(), start=1):
        if not torch.is_floating_point(base_tensor):
            merged_state[key] = base_tensor
            continue

        expert_tensors = [state.get(key, base_tensor) for state in expert_states]
        if is_target_matrix(key, base_tensor):
            merged_state[key] = diga_merge_matrix(base_tensor, expert_tensors, args.device)
        else:
            merged_state[key] = merge_non_target_tensor(base_tensor, expert_tensors)

        if idx % 50 == 0:
            print(f"  [{idx}/{len(base_state)}] {key}")

    print(f"[Save] Writing merged model to {args.output_path}")
    base_model.load_state_dict(merged_state, strict=False)
    base_model.save_pretrained(args.output_path)
    tokenizer = AutoTokenizer.from_pretrained(args.language_model_name)
    tokenizer.save_pretrained(args.output_path)
    print(f"[Done] Merged model saved to: {args.output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language_model_name", required=True, help="Base model path.")
    parser.add_argument("--adapter_paths", nargs="+", required=True,
                        help="OFT adapter paths in the desired Gram--Schmidt order.")
    parser.add_argument("--output_path", required=True, help="Merged model output directory.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    merge_models(build_parser().parse_args())


if __name__ == "__main__":
    main()
