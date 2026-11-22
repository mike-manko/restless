"""Test doubles shared by the unit tests: a chat model that plays back a script, its replies, and call log entries."""

from collections.abc import Callable, Sequence
from typing import Any, Literal

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.messages.tool import tool_call
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import Field

from restless.model.report import JudgeReport
from restless.tools.http_tool import HttpCall


class ScriptedModel(GenericFakeChatModel):
    """Plays back a fixed list of replies and remembers the messages it got. bind_tools is a no-op: the script already contains the tool calls."""

    received: list[list[BaseMessage]] = Field(default_factory=list)

    def bind_tools(self, tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool], *, tool_choice: str | None = None, **kwargs: Any) -> Runnable[LanguageModelInput, AIMessage]:
        return self

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any) -> ChatResult:
        self.received.append(list(messages))

        return super()._generate(messages, stop, run_manager, **kwargs)


def lookup(operation: str, call_id: str) -> AIMessage:
    """An executor reply that looks up one operation of the orders API."""

    return AIMessage(content="", tool_calls=[tool_call(name="get_operation", args={"api": "orders", "operation": operation}, id=call_id)])


def request(path: str, call_id: str, *, api: str = "orders") -> AIMessage:
    """An executor reply that sends one POST, to the orders API unless another API is named."""

    return AIMessage(content="", tool_calls=[tool_call(name="http_request", args={"api": api, "method": "POST", "path": path, "body": {"quantity": 2}}, id=call_id)])


def judge_reply(verdict: Literal["pass", "fail", "inconclusive"]) -> AIMessage:
    """A judge reply with this verdict, the way ScriptedModel gives structured output: a tool call named after the schema. ChatOpenAI uses a strict JSON answer instead."""

    report = {"reasoning": "Checked every expectation.", "first_failure": "", "verdict": verdict}

    return AIMessage(content="", tool_calls=[tool_call(name=JudgeReport.__name__, args=report, id="judge-1")])


def with_usage(message: AIMessage, *, input_tokens: int, output_tokens: int, cache_read: int = 0) -> AIMessage:
    """A copy of a reply that reports its token usage the way ChatOpenAI does: the cached part counted inside input_tokens."""

    usage = UsageMetadata(input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=input_tokens + output_tokens, input_token_details={"cache_read": cache_read})

    return message.model_copy(update={"usage_metadata": usage})


def http_call(*, method: str = "GET", url: str = "https://orders.test/orders", status: int | None = 200, error: str | None = None, body: Any = None) -> HttpCall:
    """One call log entry. The defaults are a quick GET that got 200."""

    return {
        "api": "orders",
        "method": method,
        "url": url,
        "request_headers": {},
        "request_body": None,
        "status": status,
        "response_headers": {},
        "body": body,
        "duration_ms": 12,
        "error": error,
    }
