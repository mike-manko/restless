"""The executor and the judge: what each one sends to the model, and how the judge reads OpenAI's answer."""

import json
from typing import Any

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableBinding
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from restless.actor.executor import Executor
from restless.actor.judge import Judge
from restless.actor.prompts import EXECUTOR_SYSTEM_PROMPT, JUDGE_SYSTEM_PROMPT, build_judge_prompt
from restless.tools.wait_tool import create_wait_tool
from tests.fakes import ScriptedModel, http_call, judge_reply


async def test_executor_sends_the_fixed_system_prompt_before_the_history() -> None:
    model = ScriptedModel(messages=iter([AIMessage(content="Done.")]))
    executor = Executor(model, [])

    reply = await executor.step([HumanMessage(content="the task")])

    assert reply.content == "Done."
    # The same system prompt in every run, so the provider's prompt cache can reuse it:
    assert model.received == [[SystemMessage(content=EXECUTOR_SYSTEM_PROMPT), HumanMessage(content="the task")]]


def test_executor_asks_for_one_tool_call_per_turn() -> None:
    model = ChatOpenAI(model_name="test-model", openai_api_key=SecretStr("sk-test"))

    executor = Executor(model, [create_wait_tool()])

    # The HTTP budget is checked after each turn, so a turn with several calls could step over it:
    assert isinstance(executor.model, RunnableBinding)
    assert executor.model.kwargs["parallel_tool_calls"] is False


async def test_judge_sees_only_the_scenario_and_the_call_log() -> None:
    model = ScriptedModel(messages=iter([judge_reply("pass")]))
    calls = [http_call(method="POST", status=201)]

    report, _ = await Judge(model).report("name: create order", calls)

    assert report.verdict == "pass"
    # Never the executor's messages: the judge judges what the APIs did, not what anyone said about it:
    assert model.received == [[SystemMessage(content=JUDGE_SYSTEM_PROMPT), HumanMessage(content=build_judge_prompt("name: create order", calls))]]


async def test_judge_asks_openai_for_strict_json_and_reads_the_verdict_from_it() -> None:
    # A fake OpenAI endpoint that records each request and answers with the verdict as JSON text:
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        answer = {"role": "assistant", "content": json.dumps({"reasoning": "Call 0 returned 201.", "first_failure": "", "verdict": "pass"})}
        return httpx.Response(200, json={"id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": "judge-model", "choices": [{"index": 0, "finish_reason": "stop", "message": answer}]})

    # The real OpenAI model class, talking to the fake endpoint:
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as openai_client:
        model = ChatOpenAI(model_name="judge-model", openai_api_key=SecretStr("sk-test"), http_async_client=openai_client)

        report, _ = await Judge(model).report("name: create order", [http_call(method="POST", status=201)])

    assert report.verdict == "pass"
    # Strict mode makes OpenAI's answer match the schema exactly, so a missing or misspelled verdict can't come back:
    assert sent[0]["response_format"]["type"] == "json_schema"
    assert sent[0]["response_format"]["json_schema"]["strict"] is True


async def test_judge_returns_its_token_usage_with_the_report() -> None:
    # A fake OpenAI endpoint that answers with a verdict and the usage block OpenAI bills by, 1,024 input tokens read from the prompt cache:
    answer = {"role": "assistant", "content": json.dumps({"reasoning": "Call 0 returned 201.", "first_failure": "", "verdict": "pass"})}
    usage = {"prompt_tokens": 1200, "completion_tokens": 80, "total_tokens": 1280, "prompt_tokens_details": {"cached_tokens": 1024}}
    completion = {"id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": "judge-model", "choices": [{"index": 0, "finish_reason": "stop", "message": answer}], "usage": usage}

    # The real OpenAI model class, talking to the fake endpoint:
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=completion))) as openai_client:
        model = ChatOpenAI(model_name="judge-model", openai_api_key=SecretStr("sk-test"), http_async_client=openai_client)

        report, judge_usage = await Judge(model).report("name: create order", [http_call(method="POST", status=201)])

    assert report.verdict == "pass"
    # The judge's own tokens come back with its verdict, the cached part counted inside the input:
    assert judge_usage is not None
    assert (judge_usage["input_tokens"], judge_usage["output_tokens"]) == (1200, 80)
    assert judge_usage.get("input_token_details", {}).get("cache_read") == 1024
