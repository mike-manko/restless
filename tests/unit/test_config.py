import pytest
from pydantic import ValidationError

from restless.config import Settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every setting from the process environment, so the developer's shell can't leak into tests.

    No yield needed: monkeypatch restores the environment itself after each test.
    """
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)


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


def test_api_key_hidden_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-very-secret")
    monkeypatch.setenv("EXECUTOR_MODEL", "test-model")
    monkeypatch.setenv("JUDGE_MODEL", "test-model")

    settings = Settings(_env_file=None)

    assert "sk-very-secret" not in repr(settings)
    assert settings.openai_api_key.get_secret_value() == "sk-very-secret"
