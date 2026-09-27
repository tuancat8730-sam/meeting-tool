"""Runtime settings, loaded from environment variables (prefix MEETING_TOOL_) or .env."""

import os
from typing import Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# LiteLLM provider prefix -> env vars that satisfy it (any one is enough)
PROVIDER_KEY_ENV: dict[str, tuple[str, ...]] = {
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
}


class MissingApiKeyError(RuntimeError):
    pass


def provider_of(model: str) -> str:
    """'gemini/gemini-2.5-flash' -> 'gemini'; bare OpenAI names ('gpt-4o') -> 'openai'."""
    return model.split("/", 1)[0] if "/" in model else "openai"


def require_api_key(model: str) -> None:
    """Fail fast with a clear message when the key for `model`'s provider is not set."""
    env_names = PROVIDER_KEY_ENV.get(provider_of(model))
    if env_names is None:
        return  # unknown provider: let LiteLLM report its own error
    if not any(os.environ.get(name) for name in env_names):
        raise MissingApiKeyError(
            f"Model '{model}' needs one of these environment variables: {', '.join(env_names)}"
        )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MEETING_TOOL_", env_file=".env", extra="ignore")

    transcribe_model: str = "gemini/gemini-2.5-flash"
    minutes_model: str = "gemini/gemini-2.5-pro"
    chunk_minutes: float = Field(default=10.0, gt=0, le=30)
    overlap_seconds: float = Field(default=60.0, ge=0)
    llm_timeout_seconds: float = Field(default=600.0, gt=0)

    @model_validator(mode="after")
    def _overlap_shorter_than_chunk(self) -> Self:
        if self.overlap_seconds >= self.chunk_minutes * 60:
            raise ValueError("overlap_seconds must be shorter than the chunk length")
        return self
