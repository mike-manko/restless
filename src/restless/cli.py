import asyncio
import io
import sys
from pathlib import Path
from typing import Annotated

import typer
from langchain_core.messages.ai import UsageMetadata
from rich.console import Console
from rich.text import Text

from restless.banner import GOBLIN_LIME, print_banner
from restless.logging_setup import configure_logging
from restless.model.report import Report
from restless.model.scenario import Scenario
from restless.runner import run_scenario
from restless.settings import get_settings

app = typer.Typer(no_args_is_help=True, add_completion=False)

# Exit codes, so a script or CI step can tell the outcomes apart. A broken setup must never look like a failing API.
# 2 is left out: Click exits with it for a usage error, like a mistyped scenario path or option:
_EXIT_CODES = {"pass": 0, "fail": 1, "inconclusive": 4}
_SETUP_ERROR = 3

# Verdict colours: a pass in the goblin's lime, a fail in red, an inconclusive result in yellow:
_VERDICT_COLOURS: dict[str, str] = {"pass": GOBLIN_LIME, "fail": "red", "inconclusive": "yellow"}


@app.callback()
def restless() -> None:
    """Run free-text API test scenarios with an LLM executor and an LLM judge."""

    # With a callback, Typer keeps "run" as a subcommand instead of making it the whole CLI, so more commands can be added later.
    # The callback runs before every command. The banner goes to stderr, so stdout carries only the report:
    print_banner(Console(stderr=True))

    # Windows encodes redirected output in its ANSI code page (cp1252), which can't hold every character a scenario or the judge writes.
    # A report that fails to print would exit with 1, the fail code, so stdout writes UTF-8:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


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

    # Rich prints the report, as it prints the banner. Click writes raw colour codes, which a Windows console without VT mode doesn't show;
    # Rich falls back to the console API there, and leaves colour out when stdout isn't a terminal. soft_wrap keeps each line whole:
    _print_report(Console(soft_wrap=True), loaded_scenario, report)

    raise typer.Exit(_EXIT_CODES[report.result.verdict])


def _print_report(console: Console, scenario: Scenario, report: Report) -> None:
    """A plain-text summary for the terminal, with the verdict in colour where the console shows colour. The markdown report comes later (design.md, build plan step 3)."""

    # Text, not str, everywhere: Rich would read [brackets] in a URL or in the judge's reasoning as markup:
    verdict = Text(report.result.verdict.upper(), style=_VERDICT_COLOURS[report.result.verdict])
    console.print(Text.assemble(f"\n{scenario.name}: ", verdict, f"  ({len(report.call_log)} HTTP calls, {report.steps} turns)\n"))

    # Every call, numbered the way the judge cites them:
    for index, call in enumerate(report.call_log):
        outcome = call["status"] if call["status"] is not None else call["error"]
        console.print(Text(f"  [{index}] {call['method']:<6} {call['url']}  -> {outcome}  ({call['duration_ms']} ms)"))

    console.print(Text(f"\n{report.result.reasoning}"))
    if report.result.first_failure:
        console.print(Text(f"\nFirst failure: {report.result.first_failure}"))

    console.print(Text(f"\n{_format_token_usage(report.executor_usage, report.judge_usage)}"))


# The run's token usage, for the executor and the judge:
def _format_token_usage(executor_usage: UsageMetadata, judge_usage: UsageMetadata | None) -> str:
    """Format the token usage for the executor and judge into a multi-line string."""

    line = "Usage statistics"
    line = line + "\n  Scenario execution:"
    line = line + f"\n  Input tokens: {executor_usage['input_tokens']} (cached: {executor_usage.get('input_token_details', {}).get('cache_read', 0)})"
    line = line + f"\n  Output tokens: {executor_usage['output_tokens']} (reasoning: {executor_usage.get('output_token_details', {}).get('reasoning', 0)})"

    if judge_usage:
        line = line + "\n\n  Result evaluation:"
        line = line + f"\n  Input tokens: {judge_usage['input_tokens']} (cached: {judge_usage.get('input_token_details', {}).get('cache_read', 0)})"
        line = line + f"\n  Output tokens: {judge_usage['output_tokens']} (reasoning: {judge_usage.get('output_token_details', {}).get('reasoning', 0)}) \n"
    else:
        line = line + "\n No result evaluation run. \n"

    return line
