from enum import Enum
from typing import Any, Dict, Optional

from fastapi import Form
from pydantic import BaseModel, Field, validator


class InputType(str, Enum):
    prompt = "prompt"
    image = "image"


class OutputType(str, Enum):
    image = "image"
    video = "video"


class GenerationParams(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=500)
    input_type: InputType = Field(default=InputType.prompt)
    output_type: OutputType = Field(default=OutputType.image)
    model_name: Optional[str] = Field(default=None)
    frames: int = Field(default=8, ge=1, le=32)
    seed: Optional[int] = Field(default=None, ge=0)

    @validator("frames")
    def _validate_frames(cls, value: int, values: Dict[str, Any]) -> int:
        if values.get("output_type") == OutputType.video and value < 2:
            raise ValueError("Video generations require at least 2 frames.")
        return value

    @classmethod
    def as_form(
        cls,
        prompt: str = Form(...),
        input_type: InputType = Form(InputType.prompt),
        output_type: OutputType = Form(OutputType.image),
        model_name: Optional[str] = Form(None),
        frames: int = Form(8),
        seed: Optional[int] = Form(None),
    ) -> "GenerationParams":
        return cls(
            prompt=prompt,
            input_type=input_type,
            output_type=output_type,
            model_name=model_name,
            frames=frames,
            seed=seed,
        )


class GenerationResponse(BaseModel):
    result_type: OutputType
    media_format: str
    encoding: str
    data: str
    metadata: Dict[str, Any]
