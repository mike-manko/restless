"""Application settings, bound from environment variables and the `.env` file.

The .NET equivalent is `IOptions<T>` bound from configuration plus user secrets:
one typed, validated object built once at startup.
"""

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from restless.model.api_config import NonEmptyStr

# Each provider's reasoning effort levels, so a level the provider doesn't have fails at startup. Which ones a model accepts depends on the model,
# and a model without reasoning rejects the parameter entirely, so None (the default) means "don't send it".
# OpenAI's, which Azure OpenAI shares:
OpenAIReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]
# Anthropic's, sent as output_config.effort. No none or minimal; max is its top level:
AnthropicReasoningEffort = Literal["low", "medium", "high", "xhigh", "max"]


class ProviderSettings(BaseModel):
    """ProviderSettings class holding what every model provider needs: an API key. Each provider adds its model fields and its own reasoning effort levels."""

    # forbid: a misspelled variable, or one that belongs to another provider (EXECUTOR__DEPLOYMENT on openai), fails at startup instead of being ignored:
    model_config = ConfigDict(extra="forbid", frozen=True)

    # SecretStr hides the value in repr() and logs; call .get_secret_value() where it is needed.
    api_key: SecretStr = Field(description="API key for this provider.")


class AnthropicModelSettings(ProviderSettings):
    """AnthropicModelSettings class describing a model called through the Anthropic API."""

    provider: Literal["anthropic"] = Field(description="Selects the Anthropic API.")
    model: NonEmptyStr = Field(description="Model id, e.g. claude-sonnet-5-5.")
    reasoning_effort: AnthropicReasoningEffort | None = Field(default=None, description="Reasoning effort. Leave empty for models without reasoning.")


class OpenAIModelSettings(ProviderSettings):
    """OpenAIModelSettings class describing a model called through the OpenAI API."""

    provider: Literal["openai"] = Field(description="Selects the OpenAI API.")
    model: NonEmptyStr = Field(description="Model id, e.g. gpt-5.")
    reasoning_effort: OpenAIReasoningEffort | None = Field(default=None, description="Reasoning effort. Leave empty for models without reasoning.")


class AzureOpenAIModelSettings(ProviderSettings):
    """AzureOpenAIModelSettings class describing a model deployed in an Azure OpenAI resource."""

    # azure_openai, not azure: LangChain's provider name, which keeps azure_ai free for Azure AI Foundry's non-OpenAI models:
    provider: Literal["azure_openai"] = Field(description="Selects Azure OpenAI.")
    endpoint: NonEmptyStr = Field(description="Resource endpoint, e.g. https://my-resource.openai.azure.com/.")
    deployment: NonEmptyStr = Field(description="Deployment name in the resource. Azure routes calls by deployment, not by model id.")
    api_version: NonEmptyStr = Field(description="Azure OpenAI API version, e.g. 2024-10-21. Newer models and reasoning effort can need a newer one.")
    reasoning_effort: OpenAIReasoningEffort | None = Field(default=None, description="Reasoning effort. Leave empty for models without reasoning.")


# The provider field picks the class: EXECUTOR__PROVIDER=azure_openai validates the other EXECUTOR__ variables as AzureOpenAIModelSettings.
# Like a JSON polymorphic type discriminator in .NET:
ModelSettings = Annotated[OpenAIModelSettings | AnthropicModelSettings | AzureOpenAIModelSettings, Field(discriminator="provider")]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # Treat `KEY=` as missing, so an empty line in .env falls back to the default
        # (or fails for a required field) instead of binding an empty string.
        env_ignore_empty=True,
        # The same environment also holds other libraries' variables (e.g. LANGSMITH_*); ignore them.
        extra="ignore",
        # Nested settings come from NAME__FIELD variables: EXECUTOR__MODEL fills settings.executor.model.
        env_nested_delimiter="__",
    )

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    http_timeout_seconds: float = Field(default=30.0, gt=0, allow_inf_nan=False)
    max_http_calls: int = Field(default=20, gt=0, description="HTTP requests the executor may send in one run.")
    default_max_turns: int = Field(default=40, gt=0, description="Executor turns in one run, operation lookups included. A scenario's max_turns overrides it.")

    # The executor (the caller) and the judge get separate models: the executor makes many short tool-calling turns,
    # the judge makes one long careful read of the call log, so they often deserve different models, effort and even providers.
    # Required fields have no default, so a missing value fails at startup, not on first use:
    executor: ModelSettings = Field(description="Model that runs the scenario and calls the tools.")
    judge: ModelSettings = Field(description="Model that judges the call log.")


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings, built on first call (like a singleton registration)."""
    return Settings()
