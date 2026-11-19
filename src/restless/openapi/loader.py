import asyncio
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import anyio
import httpx
import yaml

from restless.model.api_config import ApiConfig


class SpecLoadError(Exception):
    """Raised when an API's OpenAPI document can't be fetched or isn't an OpenAPI document."""


# Fetch every API's spec before the run starts, so a missing spec fails before any LLM call
async def load_specs(apis: Mapping[str, ApiConfig], auths: Mapping[str, httpx.Auth | None], *, base_dir: Path, timeout_seconds: float) -> dict[str, dict[str, Any]]:
    """Fetch all specs at the same time and return them as API name -> parsed document."""

    loads = [_load_spec(name, api.openapi, auth=auths.get(name), base_dir=base_dir, timeout_seconds=timeout_seconds) for name, api in apis.items()]
    specs = await asyncio.gather(*loads)

    return dict(zip(apis, specs, strict=True))


async def _load_spec(api: str, source: str, *, auth: httpx.Auth | None, base_dir: Path, timeout_seconds: float) -> dict[str, Any]:
    """Read one spec from a URL or a file, and check that it is an OpenAPI document."""

    try:
        text = await _read_source(source, auth=auth, base_dir=base_dir, timeout_seconds=timeout_seconds)
        spec = _parse(text)
    except (httpx.HTTPError, OSError, ValueError, yaml.YAMLError) as exc:
        raise SpecLoadError(f"Couldn't load the OpenAPI spec for {api!r} from {source}: {exc}") from exc

    if not isinstance(spec, dict) or "paths" not in spec:
        raise SpecLoadError(f"{source} (API {api!r}) is not an OpenAPI document: it has no 'paths'.")

    return spec


async def _read_source(source: str, *, auth: httpx.Auth | None, base_dir: Path, timeout_seconds: float) -> str:
    # Specs behind auth are common outside dev, so the request carries the same token as the API calls:
    if source.startswith(("http://", "https://")):
        async with httpx.AsyncClient(auth=auth, timeout=timeout_seconds, follow_redirects=True) as client:
            response = await client.get(source)
            response.raise_for_status()

            return response.text

    # anyio.Path reads the file without blocking the event loop (plain open() inside async code is a lint error):
    return await anyio.Path(base_dir / source).read_text(encoding="utf-8")


def _parse(text: str) -> Any:
    """JSON first, YAML otherwise. Specs come as swagger.json or openapi.yaml."""

    try:
        return json.loads(text)
    except ValueError:
        return yaml.safe_load(text)
