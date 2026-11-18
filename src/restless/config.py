"""Application settings, bound from environment variables and the `.env` file.

The .NET equivalent is `IOptions<T>` bound from configuration plus user secrets:
one typed, validated object built once at startup.
"""

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, SecretStr, StringConstraints
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # Treat `KEY=` as missing, so an empty line in .env falls back to the default
        # (or fails for a required field) instead of binding an empty string.
        env_ignore_empty=True,
        # The same environment also holds other libraries' variables (e.g. LANGSMITH_*); ignore them.
        extra="ignore",
    )

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    http_timeout_seconds: float = Field(default=30.0, gt=0, allow_inf_nan=False)

    # Required fields have no default, so a missing value fails at startup, not on first use.
    # SecretStr hides the value in repr() and logs; call .get_secret_value() where it is needed.
    openai_api_key: SecretStr
    openai_model: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings, built on first call (like a singleton registration)."""
    return Settings()
