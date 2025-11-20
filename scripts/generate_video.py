"""Generate a short video from Stable Diffusion prompts."""

from __future__ import annotations

import argparse
import os
from typing import List

import torch
from diffusers import StableDiffusionPipeline

from modules.video_generation import VideoEncodingConfig, VideoGenerationConfig, frames_to_video, generate_frames


def parse_prompts(raw: str) -> List[str]:
    return [segment.strip() for segment in raw.split("||") if segment.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", help="Prompt or prompt schedule separated by ||")
    parser.add_argument("--model", required=True, help="Path to a .safetensors model")
    parser.add_argument("--output", default="output.mp4", help="Where to write the video")
    parser.add_argument("--frames", type=int, default=16, help="Number of frames to render")
    parser.add_argument("--fps", type=int, default=8, help="Frames per second for the output video")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=20, help="Inference steps per frame")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--device", default=None, help="Execution device, e.g. cuda or cpu")
    parser.add_argument("--seed", type=int, help="Optional seed for deterministic animations")
    parser.add_argument(
        "--negative-prompt",
        dest="negative_prompt",
        help="Optional negative prompt",
    )
    args = parser.parse_args()

    prompts = parse_prompts(args.prompt)
    resolved_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    pipe = StableDiffusionPipeline.from_single_file(
        args.model,
        torch_dtype=torch.float16 if resolved_device == "cuda" else torch.float32,
        use_safetensors=True,
    )
    if resolved_device:
        pipe = pipe.to(resolved_device)

    render_config = VideoGenerationConfig(
        width=args.width,
        height=args.height,
        num_frames=args.frames,
        frame_rate=args.fps,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        device=args.device,
    )

    frames = generate_frames(
        pipe,
        prompts=prompts,
        config=render_config,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
    )
    encoding = VideoEncodingConfig(frame_rate=args.fps)
    output_path = frames_to_video(frames, args.output, encoding)
    print(f"Video written to {os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
