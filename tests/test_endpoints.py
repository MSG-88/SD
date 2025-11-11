from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from src.app import app
from src.api.deps import get_inference_service
from src.services.diffusion import ModelConfig, DiffusionManager
from src.services.inference import InferenceService


class DummyGenerator:
    def __init__(self):
        self.model_id = "dummy"
        self.config = ModelConfig(key="dummy", name="Dummy", model_id="dummy")

    @property
    def metadata(self):
        return {"key": "dummy", "name": "Dummy", "model_id": "dummy"}

    def generate_image(self, prompt: str, seed=None):
        _ = prompt, seed
        return Image.new("RGB", (64, 64), color=(0, 255, 0))


class DummyManager(DiffusionManager):
    def __init__(self):
        super().__init__({"dummy": ModelConfig(key="dummy", name="Dummy", model_id="dummy")}, "dummy")
        self._cache["dummy"] = DummyGenerator()

    def list(self):
        return [{"key": "dummy", "name": "Dummy", "model_id": "dummy"}]

    def _get_or_create(self, key: str):
        return self._cache["dummy"]


def _override_service() -> InferenceService:
    return InferenceService(DummyManager())


app.dependency_overrides[get_inference_service] = _override_service
client = TestClient(app)


def _build_image() -> BytesIO:
    image = Image.new("RGB", (64, 64), color=(255, 0, 0))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def test_prompt_to_image_generation():
    response = client.post(
        "/api/v1/generate",
        data={
            "prompt": "A calm lake with mountains",
            "input_type": "prompt",
            "output_type": "image",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["result_type"] == "image"
    assert payload["encoding"] == "base64"
    assert payload["metadata"]["prompt"] == "A calm lake with mountains"


def test_image_to_video_generation():
    buffer = _build_image()

    response = client.post(
        "/api/v1/generate",
        data={
            "prompt": "Animate this texture",
            "input_type": "image",
            "output_type": "video",
            "frames": 4,
        },
        files={"source": ("sample.png", buffer, "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["result_type"] == "video"
    assert payload["metadata"]["frames"] == 4
    assert payload["metadata"]["input_type"] == "image"
