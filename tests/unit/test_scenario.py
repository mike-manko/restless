from pathlib import Path

import pytest
from pydantic import ValidationError

from restless.model.scenario import Scenario

# One complete scenario file: the APIs it calls, what to do and what to expect.
SCENARIO_YAML = """\
name: order lifecycle
as: test-user-a
apis:
  orders: { url: https://orders.test/api, openapi: orders.json }
do: Create an order.
expect:
  - the order total is 20
"""


def test_scenario_reads_apis_as_and_expectations(tmp_path: Path) -> None:
    path = tmp_path / "scenario.yaml"
    path.write_text(SCENARIO_YAML, encoding="utf-8")

    scenario = Scenario.from_file(path)

    assert scenario.as_user == "test-user-a"
    assert scenario.apis["orders"].url == "https://orders.test/api"
    assert scenario.apis["orders"].token is None
    assert scenario.expect == ["the order total is 20"]
    assert "as: test-user-a" in scenario.to_prompt()


def test_scenario_rejects_misspelled_key(tmp_path: Path) -> None:
    path = tmp_path / "scenario.yaml"
    path.write_text(SCENARIO_YAML.replace("expect:", "expects:"), encoding="utf-8")

    with pytest.raises(ValidationError, match="expects"):
        Scenario.from_file(path)


def test_scenario_needs_at_least_one_api(tmp_path: Path) -> None:
    path = tmp_path / "scenario.yaml"
    path.write_text("name: no apis\ndo: Something.\nexpect:\n  - x\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="apis"):
        Scenario.from_file(path)
