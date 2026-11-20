from pathlib import Path

import pytest
from pydantic import ValidationError

from restless.config import Settings

# The settings template the README says to copy to .env:
ENV_EXAMPLE = Path(__file__).parents[2] / ".env.example"


def test_env_var_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("EXECUTOR_MODEL", "test-model")
    monkeypatch.setenv("JUDGE_MODEL", "test-model")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    # _env_file=None ignores the real .env, so the test only sees what it set.
    settings = Settings(_env_file=None)

    assert settings.log_level == "DEBUG"
    assert settings.http_timeout_seconds == 30.0


def test_missing_required_values_fail() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)

    missing = {error["loc"][0] for error in exc_info.value.errors()}
    assert missing == {"openai_api_key", "executor_model", "judge_model"}


@pytest.mark.parametrize(
    ("name", "value"),
    [("HTTP_TIMEOUT_SECONDS", "0"), ("HTTP_TIMEOUT_SECONDS", "inf"), ("MAX_HTTP_CALLS", "0"), ("EXECUTOR_MODEL", "   ")],
    ids=["zero-timeout", "endless-timeout", "no-budget", "blank-model"],
)
def test_invalid_value_fails_at_startup(monkeypatch: pytest.MonkeyPatch, name: str, value: str) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("EXECUTOR_MODEL", "test-model")
    monkeypatch.setenv("JUDGE_MODEL", "test-model")
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)

    assert exc_info.value.errors()[0]["loc"][0] == name.lower()


def test_api_key_hidden_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-very-secret")
    monkeypatch.setenv("EXECUTOR_MODEL", "test-model")
    monkeypatch.setenv("JUDGE_MODEL", "test-model")

    settings = Settings(_env_file=None)

    assert "sk-very-secret" not in repr(settings)
    assert settings.openai_api_key.get_secret_value() == "sk-very-secret"


def test_env_example_lists_every_setting() -> None:
    lines = ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()

    names = {line.partition("=")[0] for line in lines if "=" in line and not line.startswith("#")}

    assert names == {name.upper() for name in Settings.model_fields}


def test_env_example_loads_once_the_required_values_are_filled_in(tmp_path: Path) -> None:
    # Fill in the three values the README asks for and keep every other line as it is:
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    for name, value in {"OPENAI_API_KEY": "sk-test", "EXECUTOR_MODEL": "small-model", "JUDGE_MODEL": "big-model"}.items():
        text = text.replace(f"{name}=", f"{name}={value}")

    env_file = tmp_path / ".env"
    env_file.write_text(text, encoding="utf-8")

    settings = Settings(_env_file=env_file)

    # Lines left empty, like EXECUTOR_REASONING_EFFORT=, fall back to the default instead of failing validation:
    assert settings.executor_reasoning_effort is None
    assert settings.judge_model == "big-model"
