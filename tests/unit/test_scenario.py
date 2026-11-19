from pathlib import Path

import pytest
from pydantic import ValidationError

from restless.model.api_config import SuiteConfig
from restless.model.scenario import Scenario


def test_scenario_reads_as_and_expectations(tmp_path: Path) -> None:
    path = tmp_path / "scenario.yaml"
    path.write_text("name: order lifecycle\nas: test-user-a\ndo: Create an order.\nexpect:\n  - the order total is 20\n", encoding="utf-8")

    scenario = Scenario.from_file(path)

    assert scenario.as_user == "test-user-a"
    assert scenario.expect == ["the order total is 20"]
    assert "as: test-user-a" in scenario.to_prompt()


def test_scenario_rejects_misspelled_key(tmp_path: Path) -> None:
    path = tmp_path / "scenario.yaml"
    path.write_text("name: typo\ndo: Something.\nexpects:\n  - x\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="expects"):
        Scenario.from_file(path)


def test_suite_config_reads_apis(tmp_path: Path) -> None:
    path = tmp_path / "apis.yaml"
    path.write_text("apis:\n  orders: { url: https://orders.test/api, openapi: orders.json }\n", encoding="utf-8")

    config = SuiteConfig.from_file(path)

    assert config.apis["orders"].scope is None
