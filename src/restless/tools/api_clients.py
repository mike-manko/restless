"""httpx clients for the APIs under test: the half of HTTP the model never touches.

The http_request tool lets the model choose the method, path, headers and body.
Which host it reaches, how it signs in, how long it waits and which trace the
calls belong to are decided here, once per run.
"""

import secrets
import time
from collections.abc import Callable, Generator

import httpx
from azure.core.credentials import AccessToken, TokenCredential


def new_traceparent() -> str:
    """Make a W3C traceparent value for one test run.

    Every call in the run sends the same value, so a failed cross-service test
    can be found in Application Insights by a single trace ID.
    """
    trace_id = secrets.token_hex(16)  # 32 hex characters
    parent_id = secrets.token_hex(8)  # 16 hex characters
    return f"00-{trace_id}-{parent_id}-01"  # 00 is the format version, 01 means "sampled: record this trace"


class BearerTokenAuth(httpx.Auth):
    """Put `Authorization: Bearer <token>` on every request.

    `get_token` returns a current access token. For Entra ID use create_entra_auth below.

    Getting a token is a plain (sync) call, so the first fetch briefly blocks the
    event loop. That's fine for a CLI that runs one scenario at a time.
    """

    def __init__(self, get_token: Callable[[], str]) -> None:
        self._get_token = get_token

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response]:
        # httpx runs auth as a generator: change the request, yield it, and httpx sends it.
        # Schemes like Digest look at the response that comes back; a bearer token doesn't need to.
        request.headers["Authorization"] = f"Bearer {self._get_token()}"
        yield request


def create_entra_auth(credential: TokenCredential, scope: str) -> BearerTokenAuth:
    """Bearer auth for one Entra ID scope, keeping the token until shortly before it expires.

    Credentials backed by MSAL (client secret, managed identity) cache tokens themselves, but
    AzureCliCredential, which DefaultAzureCredential often ends up using on a dev machine, starts
    `az` for every call. That costs about a second per request, so the token is cached here.
    """

    token: AccessToken | None = None

    def get_token() -> str:
        nonlocal token

        # Renew a minute early, so a token can't expire between this check and the API reading it:
        if token is None or token.expires_on - 60 < time.time():
            token = credential.get_token(scope)

        return token.token

    return BearerTokenAuth(get_token)


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
