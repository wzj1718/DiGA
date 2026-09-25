"""Fixed merge operators used by the released LoRA, language and MLLM scripts.

All three pipelines use one-pass GS and fixed magnitude correction on both
T/R branches, with base + 0.2 * sum(delta) on non-target floating tensors.
"""

import os
import random
from fnmatch import fnmatch
from typing import Sequence, Tuple

import torch
from torch import Tensor


EPS = 1e-8
MATRIX_INCLUDE_PATTERNS = (
    "*.q_proj.weight", "*.k_proj.weight", "*.v_proj.weight", "*.o_proj.weight",
    "*.gate_proj.weight", "*.up_proj.weight", "*.down_proj.weight",
    "*.fc1.weight", "*.fc2.weight", "*.c_attn.weight", "*.c_proj.weight",
    "*.dense_h_to_4h.weight", "*.dense_4h_to_h.weight",
)
MATRIX_EXCLUDE_PATTERNS = (
    "*.embed_tokens.weight", "*.lm_head.weight", "*.layernorm*.weight", "*.norm.weight",
)
VISUAL_INCLUDE_PATTERNS = (
    "visual.blocks.*.attn.qkv.weight", "visual.blocks.*.attn.proj.weight",
    "visual.blocks.*.mlp.gate_proj.weight", "visual.blocks.*.mlp.up_proj.weight",
    "visual.blocks.*.mlp.down_proj.weight", "visual.merger.mlp.*.weight",
)


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


def is_target_matrix(name: str, tensor: Tensor, *, multimodal: bool = False) -> bool:
    if tensor.ndim != 2 or "layernorm" in name.lower() or "norm.weight" in name.lower():
        return False
    excludes = MATRIX_EXCLUDE_PATTERNS + (("lm_head.weight", "*norm*.weight") if multimodal else ())
    includes = MATRIX_INCLUDE_PATTERNS + (VISUAL_INCLUDE_PATTERNS if multimodal else ())
    return not any(fnmatch(name, p) for p in excludes) and any(fnmatch(name, p) for p in includes)


def normalize(vec: Tensor, eps: float = EPS) -> Tuple[Tensor, Tensor]:
    norm = torch.linalg.vector_norm(vec)
    return (torch.zeros_like(vec) if float(norm.item()) <= eps else vec / norm), norm


def decompose_matrix_tangent_delta(base_matrix: Tensor, delta_matrix: Tensor) -> Tuple[Tensor, Tensor]:
    rows, cols = base_matrix.shape
    if rows >= cols:
        basis, _ = torch.linalg.qr(base_matrix, mode="reduced")
        generator = basis.transpose(0, 1) @ delta_matrix
        tangent = basis @ (0.5 * (generator - generator.transpose(0, 1)))
    else:
        basis, _ = torch.linalg.qr(base_matrix.transpose(0, 1), mode="reduced")
        generator = delta_matrix @ basis
        tangent = (0.5 * (generator - generator.transpose(0, 1))) @ basis.transpose(0, 1)
    return tangent, delta_matrix - tangent


def orthogonalize_directions(directions: Sequence[Tensor]):
    ortho_dirs = []
    for direction in directions:
        work = direction.clone()
        for previous in ortho_dirs:
            work = work - torch.dot(work, previous) * previous
        work, norm = normalize(work)
        if float(norm.item()) <= EPS:
            # Preserve the experimental implementation's original-direction fallback.
            work, _ = normalize(direction)
        ortho_dirs.append(work)
    return ortho_dirs


def _consensus_norm(norms: Tensor, unit_dirs: Sequence[Tensor]) -> float:
    base_norm = norms.mean()
    if len(unit_dirs) <= 1:
        return float(base_norm.item())
    unit_matrix = torch.stack(unit_dirs, dim=0)
    similarities = unit_matrix @ unit_matrix.T
    mask = ~torch.eye(len(unit_dirs), dtype=torch.bool, device=similarities.device)
    agreement = ((similarities[mask].mean() + 1.0) / 2.0).clamp(0.0, 1.0)
    return float((base_norm * agreement).item())


def _merge_branch(deltas, norms, *, weight_scale):
    if not deltas or torch.all(norms <= EPS):
        return torch.zeros_like(deltas[0]) if deltas else torch.zeros(0)
    unit_dirs = [torch.zeros_like(d) if float(n.item()) <= EPS else d / n for d, n in zip(deltas, norms)]
    ortho_dirs = orthogonalize_directions(unit_dirs)
    scores = norms.float()
    score_norm = torch.linalg.vector_norm(scores)
    # Retain historical pre-normalization scales for identical rounding and EPS handling.
    weights = (weight_scale * (scores / score_norm.float())).to(device=deltas[0].device)
    merged_dir = torch.zeros_like(deltas[0])
    for weight, direction in zip(weights, ortho_dirs):
        merged_dir = merged_dir + weight * direction
    merged_dir, merged_norm = normalize(merged_dir)
    if float(merged_norm.item()) <= EPS:
        return torch.zeros_like(deltas[0])
    magnitude = _consensus_norm(norms, unit_dirs)
    magnitude *= 2.0
    return merged_dir * magnitude


def _merge_tangent(deltas, norms):
    return _merge_branch(deltas, norms, weight_scale=0.35)


def _merge_residual(deltas, norms):
    if torch.any(norms > EPS):
        return _merge_branch(deltas, norms, weight_scale=0.35 * 0.75)
    return torch.stack(deltas, dim=0).mean(dim=0)


def sum_deltas(deltas):
    merged = torch.zeros_like(deltas[0])
    for delta in deltas:
        merged = merged + delta
    return merged


def merge_matrix(base_tensor: Tensor, expert_tensors: Sequence[Tensor], device: str) -> Tensor:
    """Use the fixed one-pass GS merger on both T and R."""
    base_work = base_tensor.to(device=device, dtype=torch.float32)
    raw_deltas, tangents, residuals = [], [], []
    for expert in expert_tensors:
        delta = expert.to(device=device, dtype=torch.float32) - base_work
        tangent, residual = decompose_matrix_tangent_delta(base_work, delta)
        raw_deltas.append(delta.reshape(-1))
        tangents.append(tangent.reshape(-1))
        residuals.append(residual.reshape(-1))
    if torch.all(torch.stack([torch.linalg.vector_norm(d) for d in raw_deltas]) <= EPS):
        return base_tensor.clone()
    tangent_norms = torch.stack([torch.linalg.vector_norm(t) for t in tangents])
    if torch.all(tangent_norms <= EPS):
        # The original implementation falls back to the mean of complete updates.
        delta = torch.stack(raw_deltas, dim=0).mean(dim=0)
    else:
        tangent = _merge_tangent(tangents, tangent_norms)
        if float(torch.linalg.vector_norm(tangent).item()) <= EPS:
            return base_tensor.clone()
        residual_norms = torch.stack([torch.linalg.vector_norm(r) for r in residuals])
        delta = tangent + _merge_residual(residuals, residual_norms)
    return (base_work.reshape(-1) + delta).reshape_as(base_work).to(dtype=base_tensor.dtype).cpu()


def merge_non_target_ta(base_tensor: Tensor, expert_tensors: Sequence[Tensor], device: str) -> Tensor:
    """Non-target rule: base + 0.2 * sum(expert - base)."""
    if not expert_tensors:
        return base_tensor.clone()
    base_work = base_tensor.to(device=device, dtype=torch.float32)
    deltas = [expert.to(device=device, dtype=torch.float32) - base_work for expert in expert_tensors]
    return (base_work + 0.2 * sum_deltas(deltas)).to(dtype=base_tensor.dtype).cpu()
