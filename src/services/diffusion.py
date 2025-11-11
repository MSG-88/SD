from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

from PIL import Image

try:  # pragma: no cover - optional heavyweight dependency
    from diffusers import AutoPipelineForText2Image, DiffusionPipeline
except ImportError:  # pragma: no cover
    AutoPipelineForText2Image = None  # type: ignore[assignment]
    DiffusionPipeline = None  # type: ignore[assignment]

try:  # pragma: no cover - optional heavyweight dependency
    import torch
except ImportError:  # pragma: no cover - allows tests to run without torch installed
    torch = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_VALID_DEVICES = {"cuda", "cpu", "mps", "auto"}
_PIPELINE_MAP = {}
if DiffusionPipeline is not None:
    _PIPELINE_MAP["DiffusionPipeline"] = DiffusionPipeline
if AutoPipelineForText2Image is not None:
    _PIPELINE_MAP["AutoPipelineForText2Image"] = AutoPipelineForText2Image


@dataclass
class ModelConfig:
    key: str
    name: str
    model_id: str
    pipeline_class: str = "DiffusionPipeline"
    device: str = "auto"
    dtype_name: str = "bfloat16"
    guidance_scale: float = 7.5
    num_inference_steps: int = 30
    height: Optional[int] = None
    width: Optional[int] = None
    lora_repo: Optional[str] = None
    lora_weight_name: Optional[str] = None
    lora_adapter_name: Optional[str] = None
    lora_scale: float = 1.0


DEFAULT_MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "nsfw-gen-v2": ModelConfig(
        key="nsfw-gen-v2",
        name="Unfiltered NSFW Gen v2",
        model_id="UnfilteredAI/NSFW-gen-v2",
        pipeline_class="DiffusionPipeline",
        device="cuda",
        dtype_name="bfloat16",
        guidance_scale=7.5,
        num_inference_steps=30,
    ),
    "flux-dev": ModelConfig(
        key="flux-dev",
        name="Flux Dev + Lustly LoRA",
        model_id="black-forest-labs/FLUX.1-dev",
        pipeline_class="AutoPipelineForText2Image",
        device="cuda",
        dtype_name="bfloat16",
        guidance_scale=4.0,
        num_inference_steps=20,
        height=768,
        width=768,
        lora_repo="lustlyai/Flux_Lustly.ai_Uncensored_nsfw_v1",
        lora_weight_name="flux_lustly-ai_v1.safetensors",
        lora_adapter_name="lustly-v1",
        lora_scale=1.0,
    ),
    "flux-schnell": ModelConfig(
        key="flux-schnell",
        name="Flux Schnell",
        model_id="black-forest-labs/FLUX.1-schnell",
        pipeline_class="AutoPipelineForText2Image",
        device="cuda",
        dtype_name="bfloat16",
        guidance_scale=3.5,
        num_inference_steps=12,
        height=768,
        width=768,
    ),
}


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if torch is None:
        raise RuntimeError("PyTorch is required to resolve diffusion dtypes.")
    return getattr(torch, dtype_name, torch.float16)


def _device_available(target: str) -> bool:
    if torch is None:
        return target in {"cpu", "auto"}

    if target == "cuda":
        return torch.cuda.is_available()
    if target == "mps":
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    return True  # cpu / auto always usable


class DiffusionGenerator:
    def __init__(self, config: ModelConfig):
        self.config = config
        self._pipeline = None
        self._target_device = None

        if self.config.device not in _VALID_DEVICES:
            logger.warning("Unknown device '%s', defaulting to 'auto'.", self.config.device)
            self.config.device = "auto"

    @property
    def metadata(self) -> Dict[str, object]:
        return {
            "key": self.config.key,
            "name": self.config.name,
            "model_id": self.config.model_id,
            "pipeline_class": self.config.pipeline_class,
            "device": self._target_device or self.config.device,
            "dtype": self.config.dtype_name,
            "guidance_scale": self.config.guidance_scale,
            "steps": self.config.num_inference_steps,
            "height": self.config.height,
            "width": self.config.width,
            "lora": bool(self.config.lora_repo),
        }

    def generate_image(self, prompt: str, seed: Optional[int] = None) -> Image.Image:
        if torch is None:
            raise RuntimeError("PyTorch is required to run diffusion pipelines.")

        pipe = self._ensure_pipeline()

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self._target_device or "cpu").manual_seed(seed)

        kwargs = {
            "prompt": prompt,
            "guidance_scale": self.config.guidance_scale,
            "num_inference_steps": self.config.num_inference_steps,
        }
        if self.config.height:
            kwargs["height"] = self.config.height
        if self.config.width:
            kwargs["width"] = self.config.width
        if generator is not None:
            kwargs["generator"] = generator

        result = pipe(**kwargs)
        return result.images[0]

    def _ensure_pipeline(self):
        if torch is None:
            raise RuntimeError("PyTorch is required to run diffusion pipelines.")

        if self._pipeline is not None:
            return self._pipeline

        dtype = _resolve_dtype(self.config.dtype_name)
        kwargs = {"torch_dtype": dtype}

        pipeline_cls = _PIPELINE_MAP.get(self.config.pipeline_class)
        if pipeline_cls is None:
            raise RuntimeError(
                f"Diffusers pipeline '{self.config.pipeline_class}' is unavailable. "
                "Install the 'diffusers' package to use this preset."
            )

        pipe = pipeline_cls.from_pretrained(self.config.model_id, **kwargs)
        pipe.set_progress_bar_config(disable=True)

        if self.config.lora_repo and hasattr(pipe, "load_lora_weights"):
            pipe.load_lora_weights(
                self.config.lora_repo,
                weight_name=self.config.lora_weight_name,
                adapter_name=self.config.lora_adapter_name,
            )
            if hasattr(pipe, "set_adapters") and self.config.lora_adapter_name:
                pipe.set_adapters(
                    [self.config.lora_adapter_name],
                    adapter_weights=[self.config.lora_scale],
                )

        target_device = self._select_device()
        if target_device:
            _move_pipeline(pipe, target_device)
        self._target_device = target_device
        self._pipeline = pipe
        return self._pipeline

    def _select_device(self) -> Optional[str]:
        if torch is None:
            return "cpu"
        device = self.config.device
        if device == "auto":
            if _device_available("cuda"):
                return "cuda"
            if _device_available("mps"):
                return "mps"
            return "cpu"
        return device if _device_available(device) else "cpu"


def _move_pipeline(pipe: Any, target_device: str) -> None:
    reset_map = getattr(pipe, "reset_device_map", None)
    device_map = getattr(pipe, "hf_device_map", None)
    if callable(reset_map) and device_map:
        reset_map()
    pipe.to(target_device)


class DiffusionManager:
    def __init__(self, configs: Dict[str, ModelConfig], default_key: str):
        if default_key not in configs:
            raise ValueError(f"Default model '{default_key}' is not defined.")
        self.configs = configs
        self.default_key = default_key
        self._cache: Dict[str, DiffusionGenerator] = {}

    def list(self) -> Iterable[Dict[str, object]]:
        for cfg in self.configs.values():
            yield {
                "key": cfg.key,
                "name": cfg.name,
                "model_id": cfg.model_id,
                "pipeline_class": cfg.pipeline_class,
                "device": cfg.device,
                "dtype": cfg.dtype_name,
                "guidance_scale": cfg.guidance_scale,
                "steps": cfg.num_inference_steps,
                "height": cfg.height,
                "width": cfg.width,
                "lora": bool(cfg.lora_repo),
            }

    def get(self, key: Optional[str]) -> DiffusionGenerator:
        if key is None:
            key = self.default_key
        if key not in self.configs:
            raise ValueError(
                f"Unknown model '{key}'. Available options: {', '.join(self.configs.keys())}"
            )
        return self._get_or_create(key)

    def _get_or_create(self, key: str) -> DiffusionGenerator:
        if key not in self._cache:
            self._cache[key] = DiffusionGenerator(self.configs[key])
        return self._cache[key]
