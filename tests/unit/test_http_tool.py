"""Tests for the http_request tool. No network: httpx.MockTransport stands in for the API."""

import json
import re
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from langchain_core.messages import ToolCall, ToolMessage
from langchain_core.tools import BaseTool

from restless.tools.api_clients import BearerTokenAuth, create_api_client, new_traceparent
from restless.tools.http_tool import create_http_tool

TRACEPARENT = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"

Handler = Callable[[httpx.Request], httpx.Response]


def make_tool(handler: Handler, **options: Any) -> BaseTool:
    """The tool with one API, "orders", whose responses come from `handler`."""
    client = create_api_client(
        "https://orders.test/api",
        traceparent=TRACEPARENT,
        timeout_seconds=5,
        auth=BearerTokenAuth(lambda: "secret-token"),
        transport=httpx.MockTransport(handler),
    )
    return create_http_tool({"orders": client}, **options)


async def call(tool: BaseTool, **args: Any) -> ToolMessage:
    """Call the tool the way the tools node will: with a ToolCall, which returns a ToolMessage."""
    message = await tool.ainvoke(ToolCall(name=tool.name, args=args, id="call-1", type="tool_call"))
    assert isinstance(message, ToolMessage)
    return message


def ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={})


async def test_json_response_goes_to_the_model_and_the_log() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(201, json={"id": 7, "total": 20}, headers={"Location": "/api/orders/7"})

    message = await call(make_tool(handler), api="orders", method="POST", path="/orders", body={"productId": 3, "quantity": 2})

    assert message.text.startswith("HTTP 201 Created")
    assert "location: /api/orders/7" in message.text
    assert message.artifact["url"] == "https://orders.test/api/orders"
    assert message.artifact["request_body"] == {"productId": 3, "quantity": 2}
    assert message.artifact["status"] == 201
    assert message.artifact["body"] == {"id": 7, "total": 20}
    assert message.artifact["error"] is None


async def test_client_adds_token_and_traceparent_and_the_log_never_sees_the_token() -> None:
    received: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        received.append(request)
        return httpx.Response(200, json=[])

    # The model sends its own Authorization header, as it might after reading the spec.
    message = await call(
        make_tool(handler),
        api="orders",
        method="GET",
        path="/orders",
        headers={"Authorization": "Bearer <token>", "X-Request-Id": "abc"},
    )

    sent = received[0]
    assert sent.headers["authorization"] == "Bearer secret-token"
    assert sent.headers["traceparent"] == TRACEPARENT
    assert sent.headers["x-request-id"] == "abc"
    assert message.artifact["request_headers"] == {"X-Request-Id": "abc"}
    assert "secret-token" not in json.dumps(message.artifact)


@pytest.mark.parametrize(
    "path",
    [
        "https://evil.test/steal",  # another host
        "http://orders.test/api/orders",  # the right host, but without TLS
    ],
)
async def test_requests_that_leave_the_api_are_refused(path: str) -> None:
    received: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        received.append(request)
        return httpx.Response(200)

    message = await call(make_tool(handler), api="orders", method="GET", path=path)

    assert message.status == "error"
    assert "not part of this API" in message.text
    assert message.artifact is None
    assert received == []


async def test_unknown_api_is_refused_with_the_available_names() -> None:
    message = await call(make_tool(ok), api="billing", method="GET", path="/invoices")

    assert message.status == "error"
    assert "Available: orders" in message.text


async def test_invalid_arguments_go_back_to_the_model() -> None:
    message = await call(make_tool(ok), api="orders", method="get", path="/orders")

    assert message.status == "error"
    assert "Invalid arguments" in message.text
    assert "method" in message.text


async def test_timeout_is_logged_as_a_call_without_a_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("no response", request=request)

    message = await call(make_tool(handler), api="orders", method="GET", path="/orders/7")

    # The tool did its job; the API didn't answer. So this is a normal result, not a tool error.
    assert message.status == "success"
    assert message.text == "Request failed: timeout"
    assert message.artifact["status"] is None
    assert message.artifact["error"] == "timeout"


async def test_long_body_is_shortened_for_the_model_but_kept_whole_in_the_log() -> None:
    report = "x" * 50

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=report)

    message = await call(make_tool(handler, max_body_chars=10), api="orders", method="GET", path="/report")

    assert "x" * 11 not in message.text
    assert "40 more characters not shown" in message.text
    assert message.artifact["body"] == report


def test_traceparent_has_the_w3c_format() -> None:
    assert re.fullmatch(r"00-[0-9a-f]{32}-[0-9a-f]{16}-01", new_traceparent())
