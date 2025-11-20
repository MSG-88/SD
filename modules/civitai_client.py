"""Utilities for interacting with the Civitai API.

The helpers in this module keep the interaction surface small while providing
useful conveniences for the wider codebase:
* A minimal configuration object that can read from a config file or
  environment variables.
* Functions to fetch model metadata and download .safetensors files.
* Validation and clear error messages so callers can react appropriately.

The module intentionally avoids hard-coding API keys or paths. Users can set
``CIVITAI_API_KEY`` and ``CIVITAI_BASE_URL`` or provide a config file path via
``CIVITAI_CONFIG`` to keep secrets out of source control.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional
from urllib.parse import urlparse

import requests
from omegaconf import OmegaConf

DEFAULT_BASE_URL = "https://civitai.com/api/v1"


def _default_models_dir() -> str:
    env_override = os.environ.get("SD_MODELS_PATH") or os.environ.get("CIVITAI_MODELS_DIR")
    if env_override:
        return os.path.join(env_override, "Civitai")
    return os.path.join(os.getcwd(), "models", "Civitai")


@dataclass
class CivitaiConfig:
    """Configuration container for Civitai requests."""

    api_key: Optional[str] = None
    base_url: str = DEFAULT_BASE_URL
    models_dir: str = _default_models_dir()

    @classmethod
    def from_env(cls, config_path: Optional[str] = None) -> "CivitaiConfig":
        """Create a configuration instance from environment variables or a file.

        Args:
            config_path: Optional path to a YAML/JSON config file. If omitted, the
                ``CIVITAI_CONFIG`` environment variable will be used when set.
        """

        resolved_path = config_path or os.environ.get("CIVITAI_CONFIG")
        loaded = {}
        if resolved_path and os.path.exists(resolved_path):
            loaded = OmegaConf.to_container(OmegaConf.load(resolved_path), resolve=True)  # type: ignore[arg-type]

        return cls(
            api_key=loaded.get("api_key") or os.environ.get("CIVITAI_API_KEY"),
            base_url=loaded.get("base_url")
            or os.environ.get("CIVITAI_BASE_URL")
            or DEFAULT_BASE_URL,
            models_dir=loaded.get("models_dir")
            or os.environ.get("CIVITAI_MODELS_DIR")
            or _default_models_dir(),
        )


def _extract_model_id(model_id_or_url: str) -> str:
    """Normalize a model reference to an ID string.

    The Civitai UI typically links models at ``/models/<id>``. This helper will
    accept either the raw integer ID, a stringified ID, or a full URL and return
    the ID portion for use with the API.
    """

    parsed = urlparse(model_id_or_url)
    if parsed.scheme and parsed.netloc:
        parts = parsed.path.strip("/").split("/")
        if not parts:
            raise ValueError(f"Could not parse model id from URL: {model_id_or_url}")
        return parts[-1]

    return model_id_or_url


def _build_headers(config: CivitaiConfig) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    return headers


def fetch_model_metadata(model_id_or_url: str, config: Optional[CivitaiConfig] = None) -> Dict:
    """Fetch model metadata from the Civitai API.

    Args:
        model_id_or_url: Either the model's integer ID or a URL containing it.
        config: Optional :class:`CivitaiConfig` controlling how to talk to the API.

    Returns:
        Parsed JSON metadata.

    Raises:
        RuntimeError: When the API call fails or returns a non-200 status.
    """

    active_config = config or CivitaiConfig.from_env()
    model_id = _extract_model_id(model_id_or_url)

    url = f"{active_config.base_url.rstrip('/')}/models/{model_id}"
    response = requests.get(url, headers=_build_headers(active_config), timeout=30)
    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to fetch metadata for model {model_id}: HTTP {response.status_code}"
        )
    return response.json()


def _select_download(metadata: Dict) -> Dict:
    model_versions = metadata.get("modelVersions") or []
    if not model_versions:
        raise RuntimeError("Metadata does not contain any modelVersions entries")

    for file_info in model_versions[0].get("files", []):
        if file_info.get("name", "").endswith(".safetensors"):
            return file_info

    raise RuntimeError("No .safetensors file found in model metadata")


def download_model(
    model_id_or_url: str,
    destination_dir: Optional[str] = None,
    filename: Optional[str] = None,
    config: Optional[CivitaiConfig] = None,
) -> str:
    """Download a Stable Diffusion checkpoint from Civitai.

    Args:
        model_id_or_url: Model identifier or URL.
        destination_dir: Optional override for the download directory.
        filename: Optional override for the filename; defaults to the one provided
            by the metadata.
        config: Optional configuration for the Civitai API.

    Returns:
        The absolute path to the downloaded file.
    """

    active_config = config or CivitaiConfig.from_env()
    metadata = fetch_model_metadata(model_id_or_url, active_config)
    file_info = _select_download(metadata)

    os.makedirs(destination_dir or active_config.models_dir, exist_ok=True)
    output_name = filename or file_info.get("name") or f"{metadata.get('id', 'model')}.safetensors"
    output_path = os.path.abspath(os.path.join(destination_dir or active_config.models_dir, output_name))

    response = requests.get(
        file_info["downloadUrl"],
        headers=_build_headers(active_config),
        stream=True,
        timeout=120,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to download model {model_id_or_url}: HTTP {response.status_code}"
        )

    with open(output_path, "wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)

    return output_path
