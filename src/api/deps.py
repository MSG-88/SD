import logging
from functools import lru_cache

from src.core.config import settings
from src.services.diffusion import DEFAULT_MODEL_CONFIGS, DiffusionManager
from src.services.inference import InferenceService

logger = logging.getLogger(__name__)


@lru_cache
def get_inference_service() -> InferenceService:
    _ensure_hf_login()
    manager = DiffusionManager(DEFAULT_MODEL_CONFIGS, settings.default_model_name)
    return InferenceService(manager)


def _ensure_hf_login() -> None:
    token = settings.huggingface_token
    if not token:
        return

    try:
        from huggingface_hub import HfFolder, login
    except ImportError:  # pragma: no cover - defensive path
        logger.warning("huggingface_hub is not installed; skipping authentication.")
        return

    stored = HfFolder.get_token()
    if stored == token:
        return

    try:
        login(token=token, add_to_git_credential=False)
    except Exception as exc:  # pragma: no cover - network errors
        logger.error("Failed to authenticate with Hugging Face Hub: %s", exc)
