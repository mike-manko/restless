"""Tests for the restless command. run_scenario is replaced by a fake, so no spec, API or model is needed."""

import io
from pathlib import Path
from typing import Any, Literal

import pytest
from langchain_core.messages.ai import UsageMetadata
from pydantic import SecretStr
from rich.console import Console
from typer.testing import CliRunner, Result

from restless.cli import _print_report, app
from restless.model.report import JudgeReport, Report
from restless.model.scenario import Scenario
from restless.openapi.loader import SpecLoadError
from restless.settings import Settings
from restless.tools.http_tool import HttpCall
from tests.fakes import http_call

SCENARIO_YAML = """\
name: create order
apis:
  orders: { url: https://orders.test, openapi: orders.json }
do: Create an order.
expect:
  - the API answers 201
"""


@pytest.fixture(autouse=True)
def use_test_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings built here instead of from .env, and logging left the way pytest set it up."""

    settings = Settings(_env_file=None, openai_api_key=SecretStr("sk-test"), executor_model="test-model", judge_model="test-model")
    monkeypatch.setattr("restless.cli.get_settings", lambda: settings)
    monkeypatch.setattr("restless.cli.configure_logging", lambda level: None)


@pytest.fixture
def scenario_file(tmp_path: Path) -> Path:
    """A valid scenario file. Its spec is never read, because run_scenario is replaced."""

    path = tmp_path / "scenario.yaml"
    path.write_text(SCENARIO_YAML, encoding="utf-8")

    return path


def finished_run(verdict: Literal["pass", "fail", "inconclusive"], *, calls: list[HttpCall] | None = None, first_failure: str = "") -> Report:
    """The report of a run that reached a verdict."""

    result = JudgeReport(reasoning="The order was created but never fetched.", first_failure=first_failure, verdict=verdict)
    executor_usage = UsageMetadata(input_tokens=48210, output_tokens=2950, total_tokens=51160, input_token_details={"cache_read": 41600})
    judge_usage = UsageMetadata(input_tokens=9800, output_tokens=1200, total_tokens=11000)

    return Report(result=result, call_log=calls or [], hard_failures=[], steps=3, executor_usage=executor_usage, judge_usage=judge_usage)


def replace_run(monkeypatch: pytest.MonkeyPatch, report: Report) -> list[dict[str, Any]]:
    """Make the command call a fake run_scenario that returns `report`. Returns the list the fake adds its arguments to, one entry per call."""

    calls: list[dict[str, Any]] = []

    async def fake_run_scenario(scenario: Scenario, *, base_dir: Path, settings: Settings, token: str | None = None) -> Report:
        calls.append({"base_dir": base_dir, "token": token})

        return report

    monkeypatch.setattr("restless.cli.run_scenario", fake_run_scenario)

    return calls


def run_command(*args: str, env: dict[str, str] | None = None, charset: str = "utf-8") -> Result:
    """Run the restless command with its output in `charset`. A crash fails the test, instead of turning into exit code 1, which is also the fail verdict's code."""

    return CliRunner(charset=charset).invoke(app, list(args), env=env, catch_exceptions=False)


@pytest.mark.parametrize(("verdict", "exit_code"), [("pass", 0), ("fail", 1), ("inconclusive", 4)])
def test_verdict_becomes_the_exit_code(monkeypatch: pytest.MonkeyPatch, scenario_file: Path, verdict: Literal["pass", "fail", "inconclusive"], exit_code: int) -> None:
    replace_run(monkeypatch, finished_run(verdict))

    result = run_command("run", str(scenario_file))

    assert result.exit_code == exit_code


@pytest.mark.parametrize(
    ("verdict", "colour"),
    [("pass", "\x1b[38;2;170;240;40m"), ("fail", "\x1b[31m"), ("inconclusive", "\x1b[33m")],
    ids=["pass-in-goblin-lime", "fail-in-red", "inconclusive-in-yellow"],
)
def test_verdict_is_coloured_in_a_terminal(scenario_file: Path, verdict: Literal["pass", "fail", "inconclusive"], colour: str) -> None:
    # A terminal that shows full colour, as Rich sees one with VT mode on:
    output = io.StringIO()
    console = Console(file=output, force_terminal=True, color_system="truecolor", legacy_windows=False, soft_wrap=True)

    _print_report(console, Scenario.from_file(scenario_file), finished_run(verdict))

    # Only the verdict takes the colour; the name and the counts stay plain. The goblin's lime, #aaf028, is 170, 240, 40 in RGB:
    assert f"create order: {colour}{verdict.upper()}\x1b[0m  (0 HTTP calls, 3 turns)" in output.getvalue()


def test_report_lists_every_call_the_way_the_judge_numbers_them(monkeypatch: pytest.MonkeyPatch, scenario_file: Path) -> None:
    calls = [http_call(method="POST", status=201), http_call(status=None, error="timeout")]
    replace_run(monkeypatch, finished_run("fail", calls=calls, first_failure="call 1: GET /orders timed out"))

    result = run_command("run", str(scenario_file))

    lines = result.stdout.splitlines()
    assert "create order: FAIL  (2 HTTP calls, 3 turns)" in lines
    # Numbered from 0 like the judge's call indexes, with the error where no status came back:
    assert "  [0] POST   https://orders.test/orders  -> 201  (12 ms)" in lines
    assert "  [1] GET    https://orders.test/orders  -> timeout  (12 ms)" in lines
    assert "The order was created but never fetched." in lines
    assert "First failure: call 1: GET /orders timed out" in lines


def test_report_is_written_as_utf8_whatever_the_output_encoding(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # A scenario name with a character that cp1252, the Windows encoding for redirected output, doesn't have:
    path = tmp_path / "scenario.yaml"
    path.write_text(SCENARIO_YAML.replace("name: create order", "name: order → invoice"), encoding="utf-8")
    replace_run(monkeypatch, finished_run("pass"))

    result = run_command("run", str(path), charset="cp1252")

    # The pass keeps its exit code, instead of crashing with 1, the fail code:
    assert result.exit_code == 0
    assert "order → invoice: PASS".encode() in result.stdout_bytes


def test_setup_error_gets_its_own_exit_code_and_goes_to_stderr(monkeypatch: pytest.MonkeyPatch, scenario_file: Path) -> None:
    # A spec that can't be loaded, the most common broken setup:
    async def failing_run_scenario(scenario: Scenario, **options: Any) -> Report:
        raise SpecLoadError("Couldn't load the OpenAPI spec for 'orders' from orders.json")

    monkeypatch.setattr("restless.cli.run_scenario", failing_run_scenario)

    result = run_command("run", str(scenario_file))

    # A broken setup must never look like a failing API, and stdout stays free for the report:
    assert result.exit_code == 3
    assert "Setup error, the test did not run to a verdict: SpecLoadError" in result.stderr
    assert result.stdout == ""


def test_usage_error_does_not_share_an_exit_code_with_a_verdict(tmp_path: Path) -> None:
    result = run_command("run", str(tmp_path / "missing.yaml"))

    # Click's own code for a mistyped path or option. No verdict uses it, so a CI step can't take it for one:
    assert result.exit_code == 2
    assert result.stderr.startswith("Usage:")


def test_relative_spec_paths_start_from_the_scenario_files_folder(monkeypatch: pytest.MonkeyPatch, scenario_file: Path) -> None:
    calls = replace_run(monkeypatch, finished_run("pass"))

    run_command("run", str(scenario_file))

    # Not the folder the command runs in, which is the repository root here:
    assert calls[0]["base_dir"] == scenario_file.parent


@pytest.mark.parametrize(
    ("args", "env", "token"),
    [
        (["--token", "cli-token"], {}, "cli-token"),
        ([], {"RESTLESS_TOKEN": "env-token"}, "env-token"),
        ([], {}, None),
    ],
    ids=["option", "environment", "none"],
)
def test_token_comes_from_the_option_or_the_environment(monkeypatch: pytest.MonkeyPatch, scenario_file: Path, args: list[str], env: dict[str, str], token: str | None) -> None:
    calls = replace_run(monkeypatch, finished_run("pass"))

    run_command("run", str(scenario_file), *args, env=env)

    assert calls[0]["token"] == token
