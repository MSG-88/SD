from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application level configuration."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    project_name: str = Field(default="Diffusion Image API")
    api_prefix: str = Field(default="/api/v1")
    data_dir: Path = Field(default=Path("data"))

    default_model_name: str = Field(default="nsfw-gen-v2")
    # Read the Hugging Face token from environment (preferred) or a .env file.
    # IMPORTANT: Do NOT commit real tokens to source control. Use .env or environment
    # variables like HUGGINGFACEHUB_API_TOKEN, HF_TOKEN, or HUGGINGFACE_TOKEN.
    huggingface_token: str | None = Field(
        default=None,
        validation_alias=AliasChoices("HUGGINGFACE_TOKEN", "huggingface_token"),
    )


settings = Settings()
