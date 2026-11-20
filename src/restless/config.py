"""Application settings, bound from environment variables and the `.env` file.

The .NET equivalent is `IOptions<T>` bound from configuration plus user secrets:
one typed, validated object built once at startup.
"""

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, SecretStr, StringConstraints
from pydantic_settings import BaseSettings, SettingsConfigDict

ModelName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

# OpenAI's reasoning_effort values. Which ones a model accepts depends on the model, and a model without
# reasoning rejects the parameter entirely, so None (the default) means "don't send it":
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]


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
    max_http_calls: int = Field(default=20, gt=0, description="HTTP requests the executor may send in one run.")
    max_turns: int = Field(default=40, gt=0, description="Executor turns in one run, operation lookups included.")

    # Required fields have no default, so a missing value fails at startup, not on first use.
    # SecretStr hides the value in repr() and logs; call .get_secret_value() where it is needed.
    openai_api_key: SecretStr

    # The executor (the caller) and the judge get separate models: the executor makes many short tool-calling turns,
    # the judge makes one long careful read of the call log, so they often deserve different models and effort:
    executor_model: ModelName = Field(description="Model that runs the scenario and calls the tools.")
    executor_reasoning_effort: ReasoningEffort | None = Field(default=None, description="Reasoning effort for the executor. Leave empty for models without reasoning.")
    judge_model: ModelName = Field(description="Model that judges the call log.")
    judge_reasoning_effort: ReasoningEffort | None = Field(default=None, description="Reasoning effort for the judge. Leave empty for models without reasoning.")


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings, built on first call (like a singleton registration)."""
    return Settings()
