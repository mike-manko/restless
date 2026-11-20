import httpx
import pytest
from openai.lib._pydantic import to_strict_json_schema
from pydantic import SecretStr, ValidationError

from restless.config import Settings
from restless.model.api_config import ApiConfig
from restless.model.report import JudgeReport
from restless.runner import _create_auths, _create_model

APIS = {
    "orders": ApiConfig(url="https://orders.test", openapi="orders.json", token=SecretStr("from-config")),
    "billing": ApiConfig(url="https://billing.test", openapi="billing.json"),
}


def sent_authorization(auth: httpx.Auth | None) -> str | None:
    """The Authorization header this auth puts on a request, or None for no auth."""

    if auth is None:
        return None

    request = next(auth.auth_flow(httpx.Request("GET", "https://any.test")))

    return request.headers["Authorization"]


def test_config_token_is_used_and_api_without_token_gets_no_auth() -> None:
    auths = _create_auths(APIS, token=None)

    assert sent_authorization(auths["orders"]) == "Bearer from-config"
    assert sent_authorization(auths["billing"]) is None


def test_token_parameter_overrides_config_for_every_api() -> None:
    auths = _create_auths(APIS, token="from-cli")

    assert sent_authorization(auths["orders"]) == "Bearer from-cli"
    assert sent_authorization(auths["billing"]) == "Bearer from-cli"


def test_token_hidden_in_config_repr() -> None:
    assert "from-config" not in repr(APIS["orders"])


def test_each_role_gets_its_own_model_and_reasoning_effort() -> None:
    settings = Settings(_env_file=None, openai_api_key=SecretStr("sk-test"), executor_model="small-model", judge_model="big-model", judge_reasoning_effort="high")

    executor = _create_model(settings.executor_model, settings.executor_reasoning_effort, settings)
    judge = _create_model(settings.judge_model, settings.judge_reasoning_effort, settings)

    assert (executor.model_name, executor.reasoning_effort) == ("small-model", None)
    assert (judge.model_name, judge.reasoning_effort) == ("big-model", "high")


def test_unknown_reasoning_effort_fails_at_startup() -> None:
    with pytest.raises(ValidationError, match="judge_reasoning_effort"):
        Settings(_env_file=None, openai_api_key=SecretStr("sk-test"), executor_model="m", judge_model="m", judge_reasoning_effort="extreme")  # type: ignore[arg-type]


def test_judge_schema_has_no_defaults_for_openai_strict_mode() -> None:
    # OpenAI's strict structured output rejects "default" anywhere in the schema, and needs every field required:
    schema = to_strict_json_schema(JudgeReport)

    assert all("default" not in field for field in schema["properties"].values())
    assert set(schema["required"]) == {"reasoning", "first_failure", "verdict"}
