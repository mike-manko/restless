from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from restless.settings import AzureOpenAIModelSettings, OpenAIModelSettings, Settings

# The settings template the README says to copy to .env:
ENV_EXAMPLE = Path(__file__).parents[2] / ".env.example"

# The settings fields that hold a model, each read from NAME__FIELD variables:
MODEL_ROLES = ("executor", "judge")

# The smallest environment that loads: both roles on OpenAI.
OPENAI_ENV = {
    "EXECUTOR__PROVIDER": "openai",
    "EXECUTOR__MODEL": "test-model",
    "EXECUTOR__API_KEY": "sk-test",
    "JUDGE__PROVIDER": "openai",
    "JUDGE__MODEL": "test-model",
    "JUDGE__API_KEY": "sk-test",
}


def set_env(monkeypatch: pytest.MonkeyPatch, values: dict[str, str]) -> None:
    """Set each variable in the process environment for this test."""

    for name, value in values.items():
        monkeypatch.setenv(name, value)


def test_env_var_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    set_env(monkeypatch, OPENAI_ENV)
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    # _env_file=None ignores the real .env, so the test only sees what it set.
    settings = Settings(_env_file=None)

    assert settings.log_level == "DEBUG"
    assert settings.http_timeout_seconds == 30.0


def test_missing_required_values_fail() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)

    missing = {error["loc"][0] for error in exc_info.value.errors()}
    assert missing == {"executor", "judge"}


def test_each_role_picks_its_own_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    set_env(monkeypatch, OPENAI_ENV)

    # Move the executor to Azure: the OpenAI-only MODEL line goes, the Azure lines come in:
    monkeypatch.delenv("EXECUTOR__MODEL")
    set_env(
        monkeypatch,
        {
            "EXECUTOR__PROVIDER": "azure_openai",
            "EXECUTOR__ENDPOINT": "https://my-resource.openai.azure.com/",
            "EXECUTOR__DEPLOYMENT": "gpt-deployment",
            "EXECUTOR__API_VERSION": "2024-10-21",
        },
    )

    settings = Settings(_env_file=None)

    assert settings.executor == AzureOpenAIModelSettings(
        provider="azure_openai", endpoint="https://my-resource.openai.azure.com/", deployment="gpt-deployment", api_version="2024-10-21", api_key=SecretStr("sk-test")
    )
    assert isinstance(settings.judge, OpenAIModelSettings)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("HTTP_TIMEOUT_SECONDS", "0"),
        ("HTTP_TIMEOUT_SECONDS", "inf"),
        ("MAX_HTTP_CALLS", "0"),
        ("EXECUTOR__MODEL", "   "),
        ("JUDGE__REASONING_EFFORT", "extreme"),
        ("EXECUTOR__MODLE", "test-model"),
        ("EXECUTOR__DEPLOYMENT", "gpt-deployment"),
    ],
    ids=["zero-timeout", "endless-timeout", "no-budget", "blank-model", "unknown-effort", "misspelled-name", "other-providers-setting"],
)
def test_invalid_value_fails_at_startup(monkeypatch: pytest.MonkeyPatch, name: str, value: str) -> None:
    set_env(monkeypatch, OPENAI_ENV)
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)

    # The error points at the variable: ("executor", "openai", "model") is EXECUTOR__MODEL, without the provider tag in between:
    loc = exc_info.value.errors()[0]["loc"]
    assert "__".join(str(part) for part in (loc[0], *loc[2:])).upper() == name


def test_unknown_provider_fails_with_the_known_ones(monkeypatch: pytest.MonkeyPatch) -> None:
    set_env(monkeypatch, OPENAI_ENV)
    monkeypatch.setenv("EXECUTOR__PROVIDER", "azure")

    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)

    # The message names every provider, so the typo is easy to fix. Checked one by one, because pydantic lists them in union order:
    assert all(f"'{provider}'" in str(exc_info.value) for provider in ("openai", "anthropic", "azure_openai"))


def test_effort_level_the_provider_lacks_fails_at_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    set_env(monkeypatch, OPENAI_ENV)

    # minimal is an OpenAI level; Anthropic has none like it:
    set_env(monkeypatch, {"EXECUTOR__PROVIDER": "anthropic", "EXECUTOR__REASONING_EFFORT": "minimal"})

    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)

    assert exc_info.value.errors()[0]["loc"] == ("executor", "anthropic", "reasoning_effort")


def test_api_key_hidden_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    set_env(monkeypatch, OPENAI_ENV)
    monkeypatch.setenv("EXECUTOR__API_KEY", "sk-very-secret")

    settings = Settings(_env_file=None)

    assert "sk-very-secret" not in repr(settings)
    assert settings.executor.api_key.get_secret_value() == "sk-very-secret"


def test_env_example_lists_every_setting() -> None:
    lines = ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()

    names = {line.partition("=")[0] for line in lines if "=" in line and not line.startswith("#")}

    # Each model role lists the settings of every provider, so switching provider means filling in lines, not looking up names:
    provider_fields = {*OpenAIModelSettings.model_fields, *AzureOpenAIModelSettings.model_fields}
    expected = {name.upper() for name in Settings.model_fields if name not in MODEL_ROLES}
    expected |= {f"{role}__{field}".upper() for role in MODEL_ROLES for field in provider_fields}
    assert names == expected


def test_env_example_loads_once_the_required_values_are_filled_in(tmp_path: Path) -> None:
    # Fill in the four values the README asks for and keep every other line as it is:
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    for name, value in {"EXECUTOR__MODEL": "small-model", "EXECUTOR__API_KEY": "sk-test", "JUDGE__MODEL": "big-model", "JUDGE__API_KEY": "sk-test"}.items():
        text = text.replace(f"{name}=", f"{name}={value}")

    env_file = tmp_path / ".env"
    env_file.write_text(text, encoding="utf-8")

    settings = Settings(_env_file=env_file)

    # Lines left empty, like the Azure ones and EXECUTOR__REASONING_EFFORT=, fall back to the default or drop out instead of failing validation:
    assert settings.executor == OpenAIModelSettings(provider="openai", model="small-model", api_key=SecretStr("sk-test"))
    assert settings.judge == OpenAIModelSettings(provider="openai", model="big-model", api_key=SecretStr("sk-test"))
