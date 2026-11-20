import asyncio
from pathlib import Path
from typing import Annotated

import typer

from restless.config import get_settings
from restless.logging_setup import configure_logging
from restless.model.report import Report
from restless.model.scenario import Scenario
from restless.runner import run_scenario

app = typer.Typer(no_args_is_help=True, add_completion=False)

# Exit codes, so a script or CI step can tell the outcomes apart. A broken setup must never look like a failing API:
_EXIT_CODES = {"pass": 0, "fail": 1, "inconclusive": 2}
_SETUP_ERROR = 3


@app.callback()
def restless() -> None:
    """Run free-text API test scenarios with an LLM executor and an LLM judge."""

    # With a callback, Typer keeps "run" as a subcommand instead of making it the whole CLI, so more commands can be added later.


@app.command()
def run(
    scenario: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Scenario YAML file, with its APIs.")],
    # envvar: the token can come from RESTLESS_TOKEN instead, so it doesn't end up in the shell history:
    token: Annotated[str | None, typer.Option("--token", "-t", envvar="RESTLESS_TOKEN", help="Bearer token for every API. Overrides the tokens in the scenario file.")] = None,
) -> None:
    """Run one scenario and print the verdict."""

    # Everything that isn't the API's behaviour (settings, scenario file, specs, OpenAI) is a setup error, with its own exit code:
    try:
        settings = get_settings()
        configure_logging(settings.log_level)

        loaded_scenario = Scenario.from_file(scenario)
        report = asyncio.run(run_scenario(loaded_scenario, base_dir=scenario.parent, settings=settings, token=token))
    except Exception as e:
        typer.echo(f"Setup error, the test did not run to a verdict: {type(e).__name__}: {e}", err=True)
        raise typer.Exit(_SETUP_ERROR) from e

    _print_report(loaded_scenario, report)

    raise typer.Exit(_EXIT_CODES[report.result.verdict])


def _print_report(scenario: Scenario, report: Report) -> None:
    """A plain-text summary for the terminal. The markdown report comes later (design.md, build plan step 3)."""

    typer.echo(f"\n{scenario.name}: {report.result.verdict.upper()}  ({len(report.call_log)} HTTP calls, {report.steps} turns)\n")

    # Every call, numbered the way the judge cites them:
    for index, call in enumerate(report.call_log):
        outcome = call["status"] if call["status"] is not None else call["error"]
        typer.echo(f"  [{index}] {call['method']:<6} {call['url']}  -> {outcome}  ({call['duration_ms']} ms)")

    typer.echo(f"\n{report.result.reasoning}")
    if report.result.first_failure:
        typer.echo(f"\nFirst failure: {report.result.first_failure}")
