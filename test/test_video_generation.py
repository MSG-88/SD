from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image

from modules.video_generation import VideoEncodingConfig, VideoGenerationConfig, frames_to_video, generate_frames


class DummyResult:
    def __init__(self, image: Image.Image):
        self.images = [image]


class DummyPipeline:
    def __init__(self):
        self.calls: Dict[str, Any] = {}

    def __call__(self, **kwargs):
        self.calls.setdefault("prompts", []).append(kwargs["prompt"])
        color = int(255 * (len(self.calls["prompts"]) / (kwargs.get("num_frames", 10) + 1)))
        image = Image.fromarray(np.full((kwargs["height"], kwargs["width"], 3), color, dtype=np.uint8))
        return DummyResult(image)

    def to(self, device: str):
        self.calls["device"] = device
        return self


def test_generate_frames_interpolates_prompts():
    pipeline = DummyPipeline()
    config = VideoGenerationConfig(num_frames=3, width=8, height=8)
    frames = generate_frames(pipeline, prompts=["first", "second"], config=config)

    assert len(frames) == 3
    assert pipeline.calls["prompts"][0] == "first"
    assert "AND" in pipeline.calls["prompts"][1]
    assert pipeline.calls["prompts"][2].endswith("second")
    assert isinstance(frames[0], Image.Image)


def test_frames_to_video_writes_file(tmp_path):
    frames = [Image.new("RGB", (8, 8), color) for color in ("red", "green", "blue")]
    output = tmp_path / "out.gif"
    encoding = VideoEncodingConfig(frame_rate=2, codec=None, video_format="gif")
    path = frames_to_video(frames, str(output), encoding)

    assert Path(path).exists()
    assert Path(path).stat().st_size > 0
