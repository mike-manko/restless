"""The whole graph with a scripted model and a fake API: checks the wiring without paying for tokens."""

from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages.tool import tool_call

from restless.actor.executor import Executor
from restless.actor.judge import Judge
from restless.graph.builder import build_graph
from restless.graph.context import RunContext
from restless.graph.nodes import hard_check
from restless.model.api_config import ApiConfig
from restless.model.report import JudgeReport, Report
from restless.model.scenario import Scenario
from restless.openapi.catalog import ApiCatalog
from restless.tools.api_clients import create_api_client
from restless.tools.http_tool import HttpCall, create_http_tool
from restless.tools.operation_tool import create_operation_tool
from tests.fakes import ScriptedModel, http_call, lookup, request

SPEC: dict[str, Any] = {"paths": {"/orders": {"post": {"summary": "Create an order", "responses": {"201": {"description": "Created"}}}}}}
SCENARIO = Scenario(name="create order", apis={"orders": ApiConfig(url="https://orders.test", openapi="orders.json")}, do="Create an order.", expect=["the API answers 201"])


class FixedJudge(Judge):
    """Returns a fixed verdict and remembers what it was asked to judge."""

    def __init__(self, verdict: JudgeReport):
        self._verdict = verdict
        self.call_logs: list[list[HttpCall]] = []

    async def report(self, scenario: str, call_log: list[HttpCall]) -> JudgeReport:
        self.call_logs.append(call_log)

        return self._verdict


async def run(script: Iterator[AIMessage], status: int, *, max_http_calls: int = 5, max_turns: int = 10) -> tuple[Report, FixedJudge, ScriptedModel]:
    """Run the graph against a fake API that always answers `status`. Returns the report, the judge and the executor's model."""

    transport = httpx.MockTransport(lambda _: httpx.Response(status, json={"id": 7}))
    catalog = ApiCatalog({"orders": SPEC})
    model = ScriptedModel(messages=script)
    judge = FixedJudge(JudgeReport(verdict="pass", reasoning="ok", first_failure=""))

    async with create_api_client("https://orders.test", traceparent="00-0-0-01", timeout_seconds=5, transport=transport) as client:
        tools = [create_http_tool({"orders": client}), create_operation_tool(catalog)]
        context = RunContext(
            tools={tool.name: tool for tool in tools},
            catalog=catalog,
            executor=Executor(model, tools),
            judge=judge,
            max_http_calls=max_http_calls,
            max_turns=max_turns,
        )
        result = await build_graph().ainvoke({"scenario": SCENARIO}, context=context)

    return result["report"], judge, model


async def test_lookup_then_request_then_judge() -> None:
    script = iter([lookup("POST /orders", "call-1"), request("/orders", "call-2"), AIMessage(content="Created order 7.")])

    report, judge, model = await run(script, status=201)

    assert report.result.verdict == "pass"
    assert report.steps == 3
    # The lookup isn't API behaviour, so only the HTTP call reaches the judge:
    assert [call["status"] for call in judge.call_logs[0]] == [201]
    # The executor's last turn saw the whole history, each tool result answering its own call:
    last_turn = model.received[-1]
    assert [message.type for message in last_turn] == ["system", "human", "ai", "tool", "ai", "tool"]
    assert [message.tool_call_id for message in last_turn if isinstance(message, ToolMessage)] == ["call-1", "call-2"]


async def test_server_error_stops_the_run_without_the_judge() -> None:
    script = iter([request("/orders", "call-1"), AIMessage(content="never reached")])

    report, judge, _ = await run(script, status=500)

    assert report.result.verdict == "fail"
    assert report.hard_failures == ["POST https://orders.test/orders returned 500"]
    assert judge.call_logs == []


async def test_budget_ends_the_run_as_inconclusive() -> None:
    script = iter([request("/orders", "call-1"), request("/orders", "call-2"), AIMessage(content="never reached")])

    report, judge, _ = await run(script, status=201, max_http_calls=1)

    assert report.result.verdict == "inconclusive"
    # The second turn asked for a second request, and the budget stopped the run before it went out:
    assert report.steps == 2
    assert len(report.call_log) == 1
    assert judge.call_logs == []


async def test_run_that_uses_the_whole_budget_still_reaches_the_judge() -> None:
    script = iter([request("/orders", "call-1"), AIMessage(content="Created order 7.")])

    report, judge, _ = await run(script, status=201, max_http_calls=1)

    # The last request of the budget still gets the turn that finishes the run, so a scenario that needs exactly max_http_calls requests can pass:
    assert report.result.verdict == "pass"
    assert len(judge.call_logs[0]) == 1


async def test_turn_cap_stops_a_model_that_only_looks_things_up() -> None:
    # Lookups are free, so only the turn cap can stop this model. A generator gives a fresh message per turn:
    script = (lookup("POST /orders", f"call-{number}") for number in range(20))

    report, judge, _ = await run(script, status=201, max_turns=3)

    assert report.result.verdict == "inconclusive"
    assert report.steps == 3
    assert report.call_log == []
    assert judge.call_logs == []


async def test_unknown_tool_goes_back_to_the_model_instead_of_ending_the_run() -> None:
    unknown_tool = AIMessage(content="", tool_calls=[tool_call(name="delete_everything", args={}, id="call-1")])
    script = iter([unknown_tool, AIMessage(content="That tool doesn't exist, so I stopped.")])

    _, judge, model = await run(script, status=201)

    # The model's second turn starts from the error, which answers the call that caused it:
    error = model.received[1][-1]
    assert isinstance(error, ToolMessage)
    assert error.tool_call_id == "call-1"
    assert error.status == "error"
    assert "Unknown tool 'delete_everything'" in error.text
    # The judge still gets the run, without HTTP calls:
    assert judge.call_logs == [[]]


@pytest.mark.parametrize(
    ("call", "reason"),
    [
        (http_call(status=500), "GET https://orders.test/orders returned 500"),
        (http_call(status=None, error="timeout"), "GET https://orders.test/orders timed out"),
        # A 4xx can be the executor's own mistake, which it may correct, so the judge decides:
        (http_call(status=404), None),
        (http_call(status=201), None),
    ],
    ids=["server-error", "timeout", "client-error", "success"],
)
def test_hard_check_fails_only_server_errors_and_timeouts(call: HttpCall, reason: str | None) -> None:
    assert hard_check(call) == reason
