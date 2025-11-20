from __future__ import annotations

import math

import safetensors.torch
import torch

from modules.model_merger import ModelMergeError, blend_models, merge_models


def test_merge_models_combines_weights(tmp_path):
    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"
    safetensors.torch.save_file({"weight": torch.ones(1)}, first)
    safetensors.torch.save_file({"weight": torch.ones(1) * 3}, second)

    output = tmp_path / "merged.safetensors"
    path = merge_models([str(first), str(second)], [0.25, 0.75], str(output))
    result = safetensors.torch.load_file(path)

    assert math.isclose(result["weight"].item(), 2.5)


def test_merge_models_requires_matching_shapes(tmp_path):
    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"
    safetensors.torch.save_file({"a": torch.ones(1)}, first)
    safetensors.torch.save_file({"b": torch.ones(1)}, second)

    with torch.no_grad():
        try:
            merge_models([str(first), str(second)], [0.5, 0.5], str(tmp_path / "out.safetensors"))
        except ModelMergeError as exc:
            assert "tensor keys differ" in str(exc)
        else:
            raise AssertionError("Expected ModelMergeError")


def test_blend_models_alias(tmp_path):
    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"
    safetensors.torch.save_file({"weight": torch.zeros(1)}, first)
    safetensors.torch.save_file({"weight": torch.ones(1)}, second)

    output = tmp_path / "merged.safetensors"
    blend_models([str(first), str(second)], [1, 1], str(output))
    result = safetensors.torch.load_file(str(output))
    assert math.isclose(result["weight"].item(), 0.5)
