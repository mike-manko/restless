"""The whole graph with a scripted model and a fake API: checks the wiring without paying for tokens."""

from collections.abc import Callable, Iterator, Sequence
from typing import Any

import httpx
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages.tool import tool_call
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from restless.actor.executor import Executor
from restless.actor.judge import Judge
from restless.graph.builder import build_graph
from restless.graph.context import RunContext
from restless.model.api_config import ApiConfig
from restless.model.report import JudgeReport, Report
from restless.model.scenario import Scenario
from restless.openapi.catalog import ApiCatalog
from restless.tools.api_clients import create_api_client
from restless.tools.http_tool import HttpCall, create_http_tool
from restless.tools.operation_tool import create_operation_tool

SPEC: dict[str, Any] = {"paths": {"/orders": {"post": {"summary": "Create an order", "responses": {"201": {"description": "Created"}}}}}}
SCENARIO = Scenario(name="create order", apis={"orders": ApiConfig(url="https://orders.test", openapi="orders.json")}, do="Create an order.", expect=["the API answers 201"])


class ScriptedModel(GenericFakeChatModel):
    """Plays back a fixed list of replies. bind_tools is a no-op: the script already contains the tool calls."""

    def bind_tools(self, tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool], *, tool_choice: str | None = None, **kwargs: Any) -> Runnable[LanguageModelInput, AIMessage]:
        return self


class FixedJudge(Judge):
    """Returns a fixed verdict and remembers what it was asked to judge."""

    def __init__(self, verdict: JudgeReport):
        self._verdict = verdict
        self.call_logs: list[list[HttpCall]] = []

    async def report(self, scenario: str, call_log: list[HttpCall]) -> JudgeReport:
        self.call_logs.append(call_log)

        return self._verdict


def lookup(operation: str, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[tool_call(name="get_operation", args={"api": "orders", "operation": operation}, id=call_id)])


def request(path: str, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[tool_call(name="http_request", args={"api": "orders", "method": "POST", "path": path, "body": {"quantity": 2}}, id=call_id)])


async def run(script: Iterator[AIMessage], status: int, *, max_http_calls: int = 5) -> tuple[Report, FixedJudge]:
    """Run the graph against a fake API that always answers `status`."""

    transport = httpx.MockTransport(lambda _: httpx.Response(status, json={"id": 7}))
    judge = FixedJudge(JudgeReport(verdict="pass", reasoning="ok", first_failure=""))

    async with create_api_client("https://orders.test", traceparent="00-0-0-01", timeout_seconds=5, transport=transport) as client:
        tools = [create_http_tool({"orders": client}), create_operation_tool(ApiCatalog({"orders": SPEC}))]
        context = RunContext(
            tools={tool.name: tool for tool in tools},
            catalog=ApiCatalog({"orders": SPEC}),
            executor=Executor(ScriptedModel(messages=script), tools),
            judge=judge,
            max_http_calls=max_http_calls,
            max_turns=10,
        )
        result = await build_graph().ainvoke({"scenario": SCENARIO}, context=context)

    return result["report"], judge


async def test_lookup_then_request_then_judge() -> None:
    script = iter([lookup("POST /orders", "call-1"), request("/orders", "call-2"), AIMessage(content="Created order 7.")])

    report, judge = await run(script, status=201)

    assert report.result.verdict == "pass"
    assert report.steps == 3
    # The lookup isn't API behaviour, so only the HTTP call reaches the judge:
    assert [call["status"] for call in judge.call_logs[0]] == [201]


async def test_server_error_stops_the_run_without_the_judge() -> None:
    script = iter([request("/orders", "call-1"), AIMessage(content="never reached")])

    report, judge = await run(script, status=500)

    assert report.result.verdict == "fail"
    assert report.hard_failures == ["POST https://orders.test/orders returned 500"]
    assert judge.call_logs == []


async def test_budget_ends_the_run_as_inconclusive() -> None:
    script = iter([request("/orders", "call-1"), request("/orders", "call-2"), AIMessage(content="never reached")])

    report, judge = await run(script, status=201, max_http_calls=1)

    assert report.result.verdict == "inconclusive"
    assert len(report.call_log) == 1
    assert judge.call_logs == []
