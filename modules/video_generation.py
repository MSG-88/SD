"""Video generation helpers built around Stable Diffusion pipelines.

These utilities are intentionally small and dependency-light so they can be
used inside the existing Web UI, standalone scripts, or test environments
without pulling in the full application context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image
import torch


@dataclass
class VideoGenerationConfig:
    width: int = 512
    height: int = 512
    num_frames: int = 16
    frame_rate: int = 8
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    device: Optional[str] = None
    video_format: str = "mp4"


@dataclass
class VideoEncodingConfig:
    frame_rate: int = 8
    codec: Optional[str] = "libx264"
    video_format: str = "mp4"


def _prompt_for_frame(prompts: Sequence[str], frame_index: int, total_frames: int) -> str:
    if len(prompts) == 1:
        return prompts[0]

    if len(prompts) == 0:
        raise ValueError("At least one prompt is required")

    position = frame_index / max(total_frames - 1, 1)
    scaled = position * (len(prompts) - 1)
    lower = int(np.floor(scaled))
    upper = min(lower + 1, len(prompts) - 1)
    weight = scaled - lower
    if weight == 0:
        return prompts[lower]
    return f"({prompts[lower]})^{1 - weight} AND ({prompts[upper]})^{weight}"


def _get_generator(seed: Optional[int], device: Optional[str]):
    if seed is None:
        return None
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=resolved_device)
    generator.manual_seed(seed)
    return generator


def generate_frames(
    pipeline,
    prompts: Sequence[str],
    config: Optional[VideoGenerationConfig] = None,
    negative_prompt: Optional[str] = None,
    seed: Optional[int] = None,
) -> List[Image.Image]:
    """Generate a list of frames from a text-to-image pipeline.

    Args:
        pipeline: A callable Diffusers pipeline or compatible object returning an
            object with an ``images`` attribute containing PIL Images.
        prompts: One or more prompts. If multiple prompts are provided, the
            routine will interpolate between them across the frame sequence.
        config: Rendering configuration controlling resolution and cadence.
        negative_prompt: Optional negative prompt passed to the pipeline.
        seed: Optional seed to stabilise animation.
    """

    active_config = config or VideoGenerationConfig()
    frames: List[Image.Image] = []
    generator = _get_generator(seed, active_config.device)

    if hasattr(pipeline, "to") and active_config.device:
        pipeline = pipeline.to(active_config.device)

    for frame_index in range(active_config.num_frames):
        prompt = _prompt_for_frame(prompts, frame_index, active_config.num_frames)
        kwargs = {
            "prompt": prompt,
            "height": active_config.height,
            "width": active_config.width,
            "num_inference_steps": active_config.num_inference_steps,
            "guidance_scale": active_config.guidance_scale,
        }
        if negative_prompt is not None:
            kwargs["negative_prompt"] = negative_prompt
        if generator is not None:
            kwargs["generator"] = generator
            generator = generator.manual_seed(generator.initial_seed() + 1)

        result = pipeline(**kwargs)
        if hasattr(result, "images"):
            frame = result.images[0]
        else:
            frame = result[0]
        if not isinstance(frame, Image.Image):
            frame = Image.fromarray(np.array(frame))
        frames.append(frame.convert("RGB"))

    return frames


def frames_to_video(frames: Iterable[Image.Image], output_path: str, encoding: Optional[VideoEncodingConfig] = None) -> str:
    """Encode a sequence of PIL Images into a video file.

    Args:
        frames: Sequence of PIL Image frames.
        output_path: Destination path for the encoded video.
        encoding: Encoding options including frame rate and codec.
    """

    import imageio

    active_encoding = encoding or VideoEncodingConfig()
    frames = list(frames)
    if not frames:
        raise ValueError("No frames provided for encoding")

    writer_kwargs = {
        "fps": active_encoding.frame_rate,
        "format": active_encoding.video_format,
    }
    if active_encoding.codec:
        writer_kwargs["codec"] = active_encoding.codec

    with imageio.get_writer(output_path, **writer_kwargs) as writer:
        for frame in frames:
            writer.append_data(np.array(frame.convert("RGB")))

    return output_path
