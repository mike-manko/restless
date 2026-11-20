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
        auth=BearerTokenAuth("secret-token"),
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


async def test_redirect_is_shown_to_the_model_not_followed() -> None:
    received: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        received.append(request)
        return httpx.Response(302, headers={"Location": "https://login.test/sign-in"})

    message = await call(make_tool(handler), api="orders", method="GET", path="/orders")

    # A redirect could lead to another host, so the model gets the 302 and decides what to do with it:
    assert message.artifact["status"] == 302
    assert "location: https://login.test/sign-in" in message.text
    assert len(received) == 1


async def test_unknown_api_is_refused_with_the_available_names() -> None:
    message = await call(make_tool(ok), api="billing", method="GET", path="/invoices")

    assert message.status == "error"
    assert "Available: orders" in message.text


async def test_invalid_arguments_go_back_to_the_model() -> None:
    message = await call(make_tool(ok), api="orders", method="get", path="/orders")

    assert message.status == "error"
    assert "Invalid arguments" in message.text
    assert "method" in message.text


@pytest.mark.parametrize(
    "args",
    [
        {"headers": {"X-City": "Zürich"}},  # header values are ASCII only
        {"body": {"price": float("nan")}},  # JSON has no NaN
    ],
    ids=["non-ascii-header", "nan-in-body"],
)
async def test_request_httpx_cannot_build_goes_back_to_the_model(args: dict[str, Any]) -> None:
    message = await call(make_tool(ok), api="orders", method="POST", path="/orders", **args)

    # The model's mistake, so it gets a message it can correct instead of an exception that ends the run:
    assert message.status == "error"
    assert "Can't send this request" in message.text
    assert message.artifact is None


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


@pytest.mark.parametrize(
    ("content_type", "content", "logged_body"),
    [
        # ASP.NET Core sends its errors as problem+json:
        ("application/problem+json", b'{"title": "Not Found"}', {"title": "Not Found"}),
        # Says JSON but isn't, so the judge sees the text that really came back:
        ("application/json", b"<html>Bad gateway</html>", "<html>Bad gateway</html>"),
        # Looks like JSON, but the API didn't say it is:
        ("text/plain", b'{"id": 7}', '{"id": 7}'),
        ("application/json", b"", None),
    ],
    ids=["problem-json", "broken-json", "plain-text", "empty"],
)
async def test_logged_body_is_parsed_only_when_the_api_says_it_is_json(content_type: str, content: bytes, logged_body: Any) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=content, headers={"Content-Type": content_type})

    message = await call(make_tool(handler), api="orders", method="GET", path="/orders/7")

    assert message.artifact["body"] == logged_body


def test_traceparent_has_the_w3c_format() -> None:
    assert re.fullmatch(r"00-[0-9a-f]{32}-[0-9a-f]{16}-01", new_traceparent())
