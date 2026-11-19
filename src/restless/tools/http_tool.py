"""The `http_request` tool: how the executor talks to the APIs under test.

The model picks the API, method, path, headers and JSON body. Everything it
must not control (host, auth, timeouts, the traceparent header) is already
fixed in the httpx clients this tool is given; see api_clients.py.

Each call gives back two views of the same response:

- content: curl-style text for the executor, with long bodies cut short.
- artifact: an `HttpCall` record for the call log. It keeps the full body,
  parsed as JSON where possible, because the judge's expressions read it.

LangChain puts both on the ToolMessage, but only `content` goes to the model.
"""

import time
from collections.abc import Mapping
from typing import Any, Literal, TypedDict

import httpx
from langchain_core.tools import BaseTool, StructuredTool, ToolException
from pydantic import BaseModel, Field

HttpMethod = Literal["GET", "POST", "PUT", "PATCH", "DELETE"]

# The clients set these themselves. Models sometimes copy "Authorization: Bearer <token>"
# out of a spec, so we drop these quietly rather than fail the call over it.
_PROTECTED_HEADERS = frozenset({"authorization", "host", "traceparent"})


class HttpCall(TypedDict):
    """One entry in the call log.

    A TypedDict is a plain dict at runtime, so judge expressions like
    calls[3]["body"]["total"] work on it directly, and the checkpointer can
    save it without extra code.
    """

    api: str
    method: str
    url: str
    request_headers: dict[str, str]  # only what the model sent, so the token never reaches the log
    request_body: Any
    status: int | None  # None when no response came back
    response_headers: dict[str, str]
    body: Any  # the full response body: parsed JSON, otherwise text, None when empty
    duration_ms: int
    error: str | None  # why no response came back, e.g. "timeout"


class HttpRequestArgs(BaseModel):
    """What the model fills in. The field descriptions are sent to the model with the tool."""

    api: str = Field(description="Name of the API to call, from the list in the tool description.")
    method: HttpMethod = Field(description="HTTP method to use for the request.")
    path: str = Field(description="Path as written in the OpenAPI spec, plus any query string, e.g. /orders/7 or /orders?status=open.")
    headers: dict[str, str] | None = Field(default=None, description="Extra request headers. Don't send Authorization; it is added for you.")
    body: dict[str, Any] | list[Any] | None = Field(default=None, description="JSON request body.")


def create_http_tool(clients: Mapping[str, httpx.AsyncClient], *, max_body_chars: int = 10_000) -> BaseTool:
    """Build the http_request tool on top of one httpx client per API name.

    This is a factory rather than a module-level @tool because the tool needs
    this run's clients, and a closure is the simplest way to hand them over.
    """

    async def http_request(
        api: str,
        method: HttpMethod,
        path: str,
        headers: dict[str, str] | None = None,
        body: dict[str, Any] | list[Any] | None = None,
    ) -> tuple[str, HttpCall]:
        client = _client_for(clients, api)
        sent_headers = _without_protected_headers(headers or {})
        request = _build_request(client, method, path, sent_headers, body)

        # Start the log entry with what we're sending; the response side is filled in below.
        call: HttpCall = {
            "api": api,
            "method": method,
            "url": str(request.url),
            "request_headers": sent_headers,
            "request_body": body,
            "status": None,
            "response_headers": {},
            "body": None,
            "duration_ms": 0,
            "error": None,
        }

        started = time.perf_counter()
        try:
            response = await client.send(request)
        except httpx.RequestError as exc:
            # No response at all. That tells us something about the API (the design counts a
            # timeout as a fail), so it is logged like any other call, not raised as a tool error.
            call["duration_ms"] = _ms_since(started)
            call["error"] = _describe(exc)
            return f"Request failed: {call['error']}", call

        call["duration_ms"] = _ms_since(started)
        call["status"] = response.status_code
        call["response_headers"] = dict(response.headers)
        call["body"] = _parse_body(response)
        return _render(response, call["duration_ms"], max_body_chars), call

    return StructuredTool.from_function(
        coroutine=http_request,
        name="http_request",
        description=(
            "Send one HTTP request to an API under test and get back the status, headers and body. "
            f"APIs: {', '.join(sorted(clients))}. "
            "The base URL and authentication are added for you. Long bodies are shortened."
        ),
        args_schema=HttpRequestArgs,
        response_format="content_and_artifact",
        # These two turn the model's mistakes (an unknown API, a full URL, method "get") into a
        # message it can read and correct, instead of an exception that ends the whole run.
        handle_tool_error=True,
        handle_validation_error=lambda exc: f"Invalid arguments: {exc}",
    )


# Helpers, in the order http_request uses them.


def _client_for(clients: Mapping[str, httpx.AsyncClient], api: str) -> httpx.AsyncClient:
    client = clients.get(api)
    if client is None:
        raise ToolException(f"Unknown API {api!r}. Available: {', '.join(sorted(clients))}.")

    return client


def _without_protected_headers(headers: dict[str, str]) -> dict[str, str]:
    return {name: value for name, value in headers.items() if name.lower() not in _PROTECTED_HEADERS}


def _build_request(client: httpx.AsyncClient, method: str, path: str, headers: dict[str, str], body: Any) -> httpx.Request:
    """Build the request and refuse it if it doesn't point at the client's own API.

    We check the URL httpx is about to send, not the text the model typed. Checking
    the text would mean guessing how httpx joins it to the base URL, and a wrong
    guess is how a request ends up somewhere it shouldn't.
    """
    try:
        request = client.build_request(method, path, headers=headers, json=body)
    except httpx.InvalidURL as exc:
        raise ToolException(f"Invalid path {path!r}: {exc}") from exc

    target, home = request.url, client.base_url
    if (target.scheme, target.host, target.port) != (home.scheme, home.host, home.port):
        raise ToolException(f"{target} is not part of this API ({home}). Send a path like /orders/7, not a full URL.")

    return request


def _describe(exc: httpx.RequestError) -> str:
    # Always the same word for timeouts, so the hard checks can match on it.
    if isinstance(exc, httpx.TimeoutException):
        return "timeout"

    return f"{type(exc).__name__}: {exc}"


def _ms_since(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)


def _parse_body(response: httpx.Response) -> Any:
    """Parsed JSON if the API says the body is JSON, otherwise the text. None for an empty body."""
    if not response.content:
        return None
    # "json" also matches application/problem+json, which ASP.NET Core uses for its error responses.
    if "json" in response.headers.get("content-type", ""):
        try:
            return response.json()
        except ValueError:
            pass  # it said JSON but isn't; keep the text so the judge sees what really came back

    return response.text


def _render(response: httpx.Response, duration_ms: int, max_body_chars: int) -> str:
    """Show the response roughly the way `curl -i` does; models have read plenty of that.

        HTTP 201 Created (142 ms)
        content-type: application/json
        location: /orders/7

        {"id": 7, "total": 20}

    Every header is shown. Picking only "the useful ones" works until the model
    needs a Location or paging header we decided it didn't.
    """
    lines = [f"HTTP {response.status_code} {response.reason_phrase} ({duration_ms} ms)"]
    lines += [f"{name}: {value}" for name, value in response.headers.items()]
    lines += ["", _shorten(response.text, max_body_chars) or "(empty body)"]

    return "\n".join(lines)


def _shorten(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    hidden = len(text) - max_chars

    return f"{text[:max_chars]}\n[... {hidden} more characters not shown. Use filters or paging to get a smaller response.]"
