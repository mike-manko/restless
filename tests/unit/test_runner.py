import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage
from pydantic import SecretStr, ValidationError

from restless.config import Settings
from restless.model.api_config import ApiConfig
from restless.model.report import JudgeReport, Report
from restless.model.scenario import Scenario
from restless.runner import _create_auths, _create_model, run_scenario
from restless.tools.api_clients import create_api_client
from tests.fakes import ScriptedModel, judge_reply, lookup, request

APIS = {
    "orders": ApiConfig(url="https://orders.test", openapi="orders.json", token=SecretStr("from-config")),
    "billing": ApiConfig(url="https://billing.test", openapi="billing.json"),
}

# A scenario across two APIs, whose relative openapi paths point at spec files next to it. Both files get this small spec:
SPEC: dict[str, Any] = {"paths": {"/orders": {"post": {"summary": "Create an order", "responses": {"201": {"description": "Created"}}}}}}
SCENARIO = Scenario(
    name="order creates invoice",
    apis={"orders": ApiConfig(url="https://orders.test", openapi="orders.json"), "billing": ApiConfig(url="https://billing.test", openapi="billing.json")},
    do="Create an order, then find its invoice.",
    expect=["billing has an invoice for the order"],
)


def sent_authorization(auth: httpx.Auth | None) -> str | None:
    """The Authorization header this auth puts on a request, or None for no auth."""

    if auth is None:
        return None

    request = next(auth.auth_flow(httpx.Request("GET", "https://any.test")))

    return request.headers["Authorization"]


async def run_with_fakes(script: list[AIMessage], *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, token: str | None = None, max_turns: int = 10) -> tuple[Report, list[httpx.Request]]:
    """Run SCENARIO end to end with OpenAI and the network replaced: both models play back scripts, and every API answers 201."""

    # Each API's spec file, in the folder that relative openapi paths start from:
    for api in SCENARIO.apis.values():
        (tmp_path / api.openapi).write_text(json.dumps(SPEC), encoding="utf-8")

    # Fake APIs behind the real clients, recording every request that reaches them:
    received: list[httpx.Request] = []

    def handler(incoming: httpx.Request) -> httpx.Response:
        received.append(incoming)
        return httpx.Response(201, json={"id": 7})

    monkeypatch.setattr("restless.runner.create_api_client", lambda url, **options: create_api_client(url, transport=httpx.MockTransport(handler), **options))

    # Scripted models instead of OpenAI, picked by the model name in the settings:
    models = {"executor-model": ScriptedModel(messages=iter(script)), "judge-model": ScriptedModel(messages=iter([judge_reply("pass")]))}
    monkeypatch.setattr("restless.runner._create_model", lambda name, reasoning_effort, settings: models[name])

    settings = Settings(_env_file=None, openai_api_key=SecretStr("sk-test"), executor_model="executor-model", judge_model="judge-model", http_timeout_seconds=5, max_http_calls=5, max_turns=max_turns)
    report = await run_scenario(SCENARIO, base_dir=tmp_path, settings=settings, token=token)

    return report, received


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
    # OpenAI's strict structured output rejects "default" anywhere in the schema. Checked on the model itself,
    # because OpenAI's own schema converter quietly drops None defaults and marks every field required:
    assert all(field.is_required() for field in JudgeReport.model_fields.values())


async def test_every_call_of_a_run_carries_the_token_and_the_same_traceparent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = [request("/orders", "call-1"), request("/invoices", "call-2", api="billing"), AIMessage(content="Created the order and found its invoice.")]

    report, received = await run_with_fakes(script, tmp_path=tmp_path, monkeypatch=monkeypatch, token="cli-token")

    assert report.result.verdict == "pass"
    assert [incoming.url.host for incoming in received] == ["orders.test", "billing.test"]
    # The token from the command line reaches every API, and one trace ID ties the run together across them:
    assert {incoming.headers["Authorization"] for incoming in received} == {"Bearer cli-token"}
    assert received[0].headers["traceparent"] == received[1].headers["traceparent"]


async def test_run_that_uses_every_turn_ends_on_the_turn_cap_not_the_graph_step_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A model that only looks things up uses all 20 turns: 42 graph steps, two per turn plus the first and last node:
    script = [lookup("POST /orders", f"call-{number}") for number in range(30)]

    report, _ = await run_with_fakes(script, tmp_path=tmp_path, monkeypatch=monkeypatch, max_turns=20)

    # Our own turn cap ended the run, with a report, rather than a GraphRecursionError:
    assert report.steps == 20
    assert report.result.verdict == "inconclusive"
