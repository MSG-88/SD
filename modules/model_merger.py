"""Model merging helpers for .safetensors checkpoints."""

from __future__ import annotations

import os
from typing import Iterable, Sequence

import safetensors.torch
import torch


class ModelMergeError(RuntimeError):
    """Raised when a model merge cannot be completed."""


def _validate_inputs(model_paths: Sequence[str], weights: Sequence[float]):
    if len(model_paths) != len(weights):
        raise ModelMergeError("Number of model paths and weights must match")
    if not model_paths:
        raise ModelMergeError("At least one model is required to merge")

    total_weight = float(sum(weights))
    if total_weight <= 0:
        raise ModelMergeError("Weights must sum to a positive number")

    return [w / total_weight for w in weights]


def _load_tensors(model_path: str):
    if not os.path.exists(model_path):
        raise ModelMergeError(f"Model path does not exist: {model_path}")
    return safetensors.torch.load_file(model_path)


def merge_models(model_paths: Sequence[str], weights: Sequence[float], output_path: str) -> str:
    """Merge multiple .safetensors checkpoints into a single file.

    Args:
        model_paths: Sequence of checkpoint paths to merge.
        weights: Matching sequence of weights; they will be normalised internally.
        output_path: Destination for the merged checkpoint.

    Returns:
        The absolute output path of the merged model.
    """

    normalised_weights = _validate_inputs(model_paths, weights)

    merged_tensors = {}
    base_keys = None

    for model_path, weight in zip(model_paths, normalised_weights):
        tensors = _load_tensors(model_path)
        if base_keys is None:
            base_keys = set(tensors.keys())
        elif set(tensors.keys()) != base_keys:
            raise ModelMergeError("Model architectures do not align; tensor keys differ")

        for key, tensor in tensors.items():
            scaled = tensor.mul(weight)
            if key not in merged_tensors:
                merged_tensors[key] = scaled
            else:
                merged_tensors[key] = merged_tensors[key].add(scaled)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    safetensors.torch.save_file(merged_tensors, output_path)
    return os.path.abspath(output_path)


def blend_models(model_paths: Iterable[str], weights: Iterable[float], output_path: str) -> str:
    """Alias for :func:`merge_models` to support older naming."""

    return merge_models(list(model_paths), list(weights), output_path)
