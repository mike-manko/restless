"""httpx clients for the APIs under test: the half of HTTP the model never touches.

The http_request tool lets the model choose the method, path, headers and body.
Which host it reaches, how it signs in, how long it waits and which trace the
calls belong to are decided here, once per run.
"""

import secrets
from collections.abc import Generator

import httpx


def new_traceparent() -> str:
    """Make a W3C traceparent value for one test run.

    Every call in the run sends the same value, so a failed cross-service test
    can be found in Application Insights by a single trace ID.
    """
    trace_id = secrets.token_hex(16)  # 32 hex characters
    parent_id = secrets.token_hex(8)  # 16 hex characters
    return f"00-{trace_id}-{parent_id}-01"  # 00 is the format version, 01 means "sampled: record this trace"


class BearerTokenAuth(httpx.Auth):
    """Put `Authorization: Bearer <token>` on every request, with a token given up front.

    There is no renewal: if the token expires during a run, the API answers 401 and the run shows it.
    """

    def __init__(self, token: str) -> None:
        self._token = token

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response]:
        # httpx runs auth as a generator: change the request, yield it, and httpx sends it.
        # Schemes like Digest look at the response that comes back; a bearer token doesn't need to.
        request.headers["Authorization"] = f"Bearer {self._token}"
        yield request


def create_api_client(
    base_url: str,
    *,
    traceparent: str,
    timeout_seconds: float,
    auth: httpx.Auth | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> httpx.AsyncClient:
    """Create the client for one API. Close it when the run ends (`async with` or `aclose()`).

    Args:
        base_url: The API's server URL, e.g. https://orders.test/api. The model's paths go after it.
        traceparent: The run's trace header, from new_traceparent().
        timeout_seconds: Pass Settings.http_timeout_seconds, so there is one place to change it.
        auth: None for APIs that need no sign-in, such as a local copy.
        transport: Only for tests, which pass an httpx.MockTransport instead of the network.
    """
    return httpx.AsyncClient(
        base_url=base_url,
        headers={"traceparent": traceparent},
        timeout=timeout_seconds,
        auth=auth,
        # httpx doesn't follow redirects by default; saying it here makes it a decision, not luck.
        # A redirect could lead to another host, and the model should see the 3xx anyway.
        follow_redirects=False,
        transport=transport,
    )
