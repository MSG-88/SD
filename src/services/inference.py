from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from src.schemas.request import GenerationParams, InputType, OutputType
from src.services.diffusion import DiffusionGenerator, DiffusionManager

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    media_type: OutputType
    media_format: str
    payload: bytes
    seed: Optional[int]
    frames: int = 1
    model_metadata: Dict[str, object] | None = None


class InferenceService:
    """High-level orchestration for prompt driven generations using diffusers."""

    def __init__(self, manager: DiffusionManager):
        self.manager = manager

    def available_models(self) -> Iterable[Dict[str, object]]:
        return list(self.manager.list())

    def generate(
        self, params: GenerationParams, source_image: Optional[Image.Image]
    ) -> Dict[str, object]:
        if params.input_type == InputType.image and source_image is not None:
            logger.warning("Reference images are not yet supported; proceeding with prompt only.")

        prompt = params.prompt.strip()
        if not prompt:
            raise ValueError("Prompt must not be empty.")

        generator = self.manager.get(params.model_name)

        if params.output_type == OutputType.image:
            image = generator.generate_image(prompt, seed=params.seed)
            result = GenerationResult(
                media_type=OutputType.image,
                media_format="png",
                payload=_to_png(image),
                seed=params.seed,
                model_metadata=generator.metadata,
            )
        else:
            frames = self._generate_video_frames(generator, prompt, params)
            result = GenerationResult(
                media_type=OutputType.video,
                media_format="gif",
                payload=_to_gif(frames),
                seed=params.seed,
                frames=len(frames),
                model_metadata=generator.metadata,
            )

        response = {
            "result_type": result.media_type.value,
            "media_format": result.media_format,
            "encoding": "base64",
            "data": base64.b64encode(result.payload).decode("utf-8"),
            "metadata": {
                "model": result.model_metadata,
                "prompt": prompt,
                "input_type": params.input_type.value,
                "output_type": params.output_type.value,
                "seed": result.seed,
            },
        }

        if result.media_type == OutputType.video:
            response["metadata"]["frames"] = result.frames

        return response

    def _generate_video_frames(
        self,
        generator: DiffusionGenerator,
        prompt: str,
        params: GenerationParams,
    ) -> Iterable[Image.Image]:
        frames = []
        base_seed = params.seed or 0

        for idx in range(params.frames):
            seed = base_seed + idx if base_seed else None
            frame = generator.generate_image(prompt, seed=seed)
            frames.append(frame)

        return frames


def _to_png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _to_gif(frames: Iterable[Image.Image]) -> bytes:
    buffer = io.BytesIO()
    imageio.mimsave(buffer, [np.asarray(frame) for frame in frames], format="GIF", duration=0.12)
    return buffer.getvalue()
