import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from restless.model.api_config import ApiConfig
from restless.model.scenario import Scenario

ROOT = Path(__file__).parents[2]

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

# The example scenarios in the repository, which have to keep up with the Scenario model. Ask git rather than the folder:
# .gitignore keeps local scenarios under examples/ too, and a half-written one of those must not fail the tests:
EXAMPLES = subprocess.run(["git", "ls-files", "examples/*.yaml"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.splitlines()


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
    path.write_text("name: no apis\napis: {}\ndo: Something.\nexpect:\n  - x\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="apis"):
        Scenario.from_file(path)


@pytest.mark.parametrize(("field", "value"), [("do", "   "), ("expect", []), ("expect", ["   "])], ids=["blank-do", "no-expectations", "blank-expectation"])
def test_scenario_rejects_blank_instructions_and_expectations(field: str, value: Any) -> None:
    # Start from the valid scenario and blank out one field. A scenario without a real expectation would check nothing:
    data: dict[str, Any] = yaml.safe_load(SCENARIO_YAML)
    data[field] = value

    with pytest.raises(ValidationError) as exc_info:
        Scenario.model_validate(data)

    assert exc_info.value.errors()[0]["loc"][0] == field


def test_prompt_lists_every_expectation_and_leaves_out_a_missing_as() -> None:
    scenario = Scenario(name="create order", apis={"orders": ApiConfig(url="https://orders.test", openapi="orders.json")}, do="Create an order.", expect=["the API answers 201", "the order has an id"])

    assert scenario.to_prompt() == "name: create order\ndo: Create an order.\nexpect:\n  - the API answers 201\n  - the order has an id"


@pytest.mark.parametrize("example", EXAMPLES)
def test_example_scenario_is_valid_and_has_no_token(example: str) -> None:
    scenario = Scenario.from_file(ROOT / example)

    # The examples are committed, so a token in one of them would be a leaked secret:
    assert all(api.token is None for api in scenario.apis.values())
