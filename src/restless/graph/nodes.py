import operator
from collections.abc import Sequence
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime

from restless.actor.prompts import build_executor_task
from restless.graph.context import RunContext
from restless.model.report import JudgeReport, Report
from restless.model.scenario import Scenario
from restless.tools.http_tool import HttpCall


class AgentState(TypedDict):
    scenario: Scenario
    messages: Annotated[Sequence[BaseMessage], add_messages]
    call_log: Annotated[list[HttpCall], operator.add]
    hard_failures: Annotated[list[str], operator.add]
    steps: int
    report: Report


class GraphInput(TypedDict):
    """What graph.ainvoke takes. load_scenario builds the rest of AgentState from it."""

    scenario: Scenario


# Start the run: put the first user message in the history and set every counter, so later nodes can read state["x"] directly:
def load_scenario(state: AgentState, runtime: Runtime[RunContext]) -> dict[str, Any]:
    """Build the executor's task from the scenario and the catalog's operation list."""

    task = build_executor_task(state["scenario"].to_prompt(), runtime.context.catalog.list_operations(), runtime.context.max_http_calls)

    return {"messages": [HumanMessage(content=task)], "call_log": [], "hard_failures": [], "steps": 0}


# LLM selects api method to execute based on scenario, returns
# call details, request parameters etc and passes it down to the HTTP call executor:
async def next_executor_action(state: AgentState, runtime: Runtime[RunContext]) -> dict[str, Any]:
    """Run one executor turn: send the history to the model and append its reply to the state."""

    messages: list[BaseMessage] = list(state["messages"])
    steps = state["steps"]

    ai_message = await runtime.context.executor.step(messages)
    steps += 1

    return {"messages": [ai_message], "steps": steps}


# Route based on whether the last AI message has tool calls
def has_tool_calls(state: AgentState, runtime: Runtime[RunContext]) -> Literal["call_tools", "judge_result"]:
    """Return 'call_tools' if the executor's last message asks for tools, otherwise 'judge_result': no tool call means it finished."""

    last_message = state["messages"][-1] if state["messages"] else None

    if not isinstance(last_message, AIMessage):
        return "judge_result"

    if not last_message.tool_calls:
        return "judge_result"

    return "call_tools"


# Run the tools the last AI message asked for (http_request, get_operation, wait)
async def call_tools(state: AgentState, runtime: Runtime[RunContext]) -> dict[str, Any]:
    """Run every tool call in the executor's last message. HTTP calls go to the call log and through the hard checks."""

    last_message = state["messages"][-1] if state["messages"] else None

    # isinstance also narrows the type for mypy, so no cast is needed after it:
    if not isinstance(last_message, AIMessage):
        raise TypeError("Last message must be an AIMessage")

    call_log: list[HttpCall] = []
    new_failures: list[str] = []
    tool_messages: list[ToolMessage] = []

    # Process each tool call in the last AI message:
    for tool_call in last_message.tool_calls:
        tool = runtime.context.tools.get(tool_call["name"])

        if tool is None:
            tool_messages.append(ToolMessage(content=f"Unknown tool {tool_call['name']!r}.", tool_call_id=tool_call["id"], status="error"))
            continue

        message = await tool.ainvoke(tool_call)
        tool_messages.append(message)

        # Only http_request returns an artifact. None means a lookup, a wait or the model's mistake (bad args, foreign URL), not an API call:
        if message.artifact is not None:
            call_log.append(message.artifact)
            if (reason := hard_check(message.artifact)) is not None:
                new_failures.append(reason)

    return {"call_log": call_log, "hard_failures": new_failures, "messages": tool_messages}


# Check if the agent has exceeded its budget or encountered hard failures
def check_budget(state: AgentState, runtime: Runtime[RunContext]) -> Literal["hard_failure_result", "judge_result", "next_executor_action"]:
    """Check if the agent has exceeded its budget. Return 'judge_result' if budget is exceeded, otherwise 'next_executor_action'."""

    if state["hard_failures"]:
        return "hard_failure_result"

    # The budget counts HTTP requests, so lookups are free. The turn cap stops a model that only ever looks things up:
    if len(state["call_log"]) >= runtime.context.max_http_calls or state["steps"] >= runtime.context.max_turns:
        return "judge_result"

    return "next_executor_action"


async def judge_result(state: AgentState, runtime: Runtime[RunContext]) -> dict[str, Any]:
    """Generate a judge report for the current state of the agent."""

    failures: list[str] = list(state["hard_failures"])
    call_log: list[HttpCall] = list(state["call_log"])

    # Scenario reached call budget, return result:
    if not isinstance(state["messages"][-1], AIMessage):
        reasoning = f"Stopped by the budget ({len(call_log)} of {runtime.context.max_http_calls} HTTP calls, {state['steps']} of {runtime.context.max_turns} turns) before the executor finished."
        return {"report": Report(result=JudgeReport(verdict="inconclusive", reasoning=reasoning, first_failure=""), call_log=call_log, hard_failures=failures, steps=state["steps"])}

    # Request a judge report for the current scenario and call log:
    judge_report = await runtime.context.judge.report(state["scenario"].to_prompt(), call_log)
    report = Report(result=judge_report, call_log=call_log, hard_failures=failures, steps=state["steps"])

    return {"report": report}


def hard_failure_result(state: AgentState, runtime: Runtime[RunContext]) -> dict[str, Any]:
    """Return a report indicating a hard failure occurred."""
    return {
        "report": Report(
            # The hard checks failed the test before the judge ran. List every reason, not only the first:
            # two 500s on different endpoints tell a different story than one.
            result=JudgeReport(
                verdict="fail",
                reasoning=f"Stopped by the hard checks before the judge ran: {'; '.join(state['hard_failures'])}",
                first_failure=state["hard_failures"][0],
            ),
            call_log=state["call_log"],
            hard_failures=state["hard_failures"],
            steps=state["steps"],
        )
    }


def hard_check(call: HttpCall) -> str | None:
    """Why this call fails the test on its own, or None. Code decides this, not the model."""

    if call["error"] == "timeout":
        return f"{call['method']} {call['url']} timed out"

    if call["status"] is not None and call["status"] >= 500:
        return f"{call['method']} {call['url']} returned {call['status']}"

    return None
