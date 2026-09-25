"""Model loading, vocabulary alignment and saving for fixed language mergers."""

import argparse
import gc
import os
from collections import OrderedDict
from typing import Dict, List, Sequence

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from diga_fixed import is_target_matrix, seed_torch

EMBED_KEYS = ("model.embed_tokens.weight", "lm_head.weight")
_TOKENIZER_CACHE: Dict[str, object] = {}


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


def load_state_dict(model_path: str) -> Dict[str, Tensor]:
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="cpu", torch_dtype=torch.bfloat16
    )
    state_dict = model.state_dict()
    del model
    gc.collect()
    return state_dict


def align_expert_state_dicts(
    base_model_path: str,
    expert_paths: Sequence[str],
    base_state: Dict[str, Tensor],
) -> List[Dict[str, Tensor]]:
    base_vocab = get_tokenizer(base_model_path).get_vocab()
    experts: List[Dict[str, Tensor]] = []
    for expert_path in expert_paths:
        print(f"[Load] Expert model: {expert_path}")
        expert_state = load_state_dict(expert_path)
        expert_vocab = get_tokenizer(expert_path).get_vocab()
        # Preserve vocabulary remapping only when the embedding shapes differ.
        for key in EMBED_KEYS:
            if key in expert_state and key in base_state:
                if expert_state[key].shape != base_state[key].shape:
                    print(f"  [Align] Remapping vocab for {key}")
                    expert_state[key] = remap_embedding_to_base_vocab(
                        base_vocab, expert_vocab, expert_state[key]
                    )
        experts.append(expert_state)
    return experts


def build_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--expert_model_paths", nargs="+", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def merge_models(args, merge_matrix, merge_non_target):
    seed_torch(args.seed)
    os.makedirs(args.output_path, exist_ok=True)
    print(f"[Load] Base model: {args.base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path, device_map="cpu", torch_dtype=torch.bfloat16
    )
    base_state = base_model.state_dict()
    expert_states = align_expert_state_dicts(
        args.base_model_path, args.expert_model_paths, base_state
    )
    merged_state = OrderedDict()
    for idx, (key, base_tensor) in enumerate(base_state.items(), start=1):
        if not torch.is_floating_point(base_tensor):
            merged_state[key] = base_tensor
            continue
        experts = [state.get(key, base_tensor) for state in expert_states]
        operator = merge_matrix if is_target_matrix(key, base_tensor) else merge_non_target
        merged_state[key] = operator(base_tensor, experts, args.device)
        if idx % 50 == 0:
            print(f"  [{idx}/{len(base_state)}] {key}")
    print(f"[Save] Writing merged model to {args.output_path}")
    base_model.load_state_dict(merged_state, strict=False)
    base_model.save_pretrained(args.output_path)
    AutoTokenizer.from_pretrained(args.base_model_path).save_pretrained(args.output_path)
