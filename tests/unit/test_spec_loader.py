from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from restless.model.api_config import ApiConfig
from restless.openapi.loader import SpecLoadError, load_specs
from restless.tools.api_clients import BearerTokenAuth

# A spec served by the API itself, the way the example scenarios load theirs:
SPEC_URL = "https://orders.test/openapi/v1.json"


def send_requests_to(monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx.Request], httpx.Response]) -> None:
    """Answer the loader's HTTP requests with `handler` instead of the network. The loader builds its own client, so swap the class it builds."""

    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **options: real_client(transport=httpx.MockTransport(handler), **options))


@pytest.mark.parametrize(("file_name", "text"), [("orders.json", '{"openapi": "3.0.1", "paths": {}}'), ("orders.yaml", "openapi: 3.0.1\npaths: {}\n")])
async def test_loads_json_and_yaml_files(tmp_path: Path, file_name: str, text: str) -> None:
    (tmp_path / file_name).write_text(text, encoding="utf-8")
    apis = {"orders": ApiConfig(url="https://orders.test", openapi=file_name)}

    specs = await load_specs(apis, {}, base_dir=tmp_path, timeout_seconds=5)

    assert specs["orders"]["paths"] == {}


async def test_rejects_a_file_that_is_not_a_spec(tmp_path: Path) -> None:
    (tmp_path / "orders.json").write_text('{"hello": "world"}', encoding="utf-8")
    apis = {"orders": ApiConfig(url="https://orders.test", openapi="orders.json")}

    with pytest.raises(SpecLoadError, match="no 'paths'"):
        await load_specs(apis, {}, base_dir=tmp_path, timeout_seconds=5)


async def test_missing_file_names_the_api(tmp_path: Path) -> None:
    apis = {"orders": ApiConfig(url="https://orders.test", openapi="missing.json")}

    with pytest.raises(SpecLoadError, match="'orders'"):
        await load_specs(apis, {}, base_dir=tmp_path, timeout_seconds=5)


async def test_spec_url_is_fetched_with_the_api_token(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    received: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        received.append(request)
        return httpx.Response(200, json={"openapi": "3.0.1", "paths": {}})

    send_requests_to(monkeypatch, handler)
    apis = {"orders": ApiConfig(url="https://orders.test", openapi=SPEC_URL)}

    specs = await load_specs(apis, {"orders": BearerTokenAuth("secret-token")}, base_dir=tmp_path, timeout_seconds=5)

    assert specs["orders"]["openapi"] == "3.0.1"
    # Specs behind auth are common outside dev, so the spec request carries the API's token too:
    assert received[0].headers["Authorization"] == "Bearer secret-token"


async def test_spec_url_that_answers_with_an_error_names_the_api(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # A wrong or expired token shows up here, before any LLM call:
    send_requests_to(monkeypatch, lambda request: httpx.Response(401))
    apis = {"orders": ApiConfig(url="https://orders.test", openapi=SPEC_URL)}

    with pytest.raises(SpecLoadError, match=r"'orders'.*401"):
        await load_specs(apis, {}, base_dir=tmp_path, timeout_seconds=5)
