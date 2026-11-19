from pathlib import Path

import pytest

from restless.model.api_config import ApiConfig
from restless.openapi.loader import SpecLoadError, load_specs


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
