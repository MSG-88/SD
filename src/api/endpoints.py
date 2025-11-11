import io
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from src.api.deps import get_inference_service
from src.core.config import settings
from src.schemas.request import GenerationParams, GenerationResponse
from src.services.inference import InferenceService

router = APIRouter(prefix=settings.api_prefix, tags=["generation"])


async def _read_image(upload: UploadFile | None) -> Image.Image | None:
    if upload is None:
        return None

    contents = await upload.read()
    if not contents:
        return None

    try:
        return Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from exc


@router.post("/generate", response_model=GenerationResponse)
async def generate_media(
    params: GenerationParams = Depends(GenerationParams.as_form),
    source: UploadFile | None = File(None),
    service: InferenceService = Depends(get_inference_service),
) -> Any:
    image = await _read_image(source)

    try:
        payload = service.generate(params, image)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return JSONResponse(content=payload)


@router.get("/models")
def list_models(service: InferenceService = Depends(get_inference_service)) -> Dict[str, List[Dict[str, Any]]]:
    models = list(service.available_models())
    if not models:
        raise HTTPException(status_code=404, detail="No diffusion models configured.")
    return {"models": models}
