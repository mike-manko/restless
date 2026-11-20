<p align="center">
  <img src="docs/assets/restless.gif" alt="restless animated banner" width="250">
</p>

# restless

API tests written in plain language. Give restless an OpenAPI spec, describe what should happen, and let it run the requests and check the results.

> [!NOTE]
> This is alpha. The CLI works, but scenarios, configuration and reports can change. The judge is an LLM, and its accuracy hasn't been measured yet. Treat verdicts as evidence to review.

[Quick start](#quick-start) · [Scenarios](#write-a-scenario) · [Configuration](#configuration) · [Development](#development) · [Alpha limits](#alpha-limits-and-plans)

## What it does

Writing an API test usually means writing the requests, passing IDs between them and adding assertions. With restless, you describe the flow and the expectations in a small YAML file. The executor reads the available operations, looks up endpoint details when it needs them, and makes real HTTP requests.

A separate judge reads the scenario and the HTTP call log, without the executor's reasoning. It returns `pass`, `fail` or `inconclusive`, with an explanation of what it checked. Code handles timeouts and server errors directly, before the judge runs.

The current runner supports:

- Scenarios across one or several APIs, each with its own base URL and OpenAPI document.
- JSON or YAML specs loaded from a URL or a local file.
- Bearer authentication per API, or one token supplied for the whole run.
- A wait tool for background work, plus configurable HTTP-call and executor-turn budgets.
- A terminal report with numbered requests, response status, timing and verdict reasoning.
- A shared `traceparent` header across the run's API requests.

Python 3.13+, Typer, LangGraph, OpenAI, HTTPX and Pydantic make up the runner.

## Quick start

You'll need Git, Python 3.13+ and [uv](https://docs.astral.sh/uv/). Running a scenario also needs an OpenAI API key and access to the models you configure. LLM calls cost money.

```sh
git clone https://github.com/mike-manko/restless.git
cd restless
uv sync --locked
```

Copy the example settings:

```sh
# macOS / Linux
cp .env.example .env
```

```powershell
# Windows PowerShell
Copy-Item .env.example .env
```

Edit `.env` and fill in these three required values:

```dotenv
OPENAI_API_KEY=your-api-key
EXECUTOR_MODEL=your-executor-model-id
JUDGE_MODEL=your-judge-model-id
```

Replace the placeholders with model IDs available to your account. The executor needs tool calling; the judge needs structured output. You can use the same model for both. Leave the reasoning-effort fields empty unless your chosen models support them.

Run the included scenario from the repository root:

```sh
uv run restless run examples/petstore/pet-lifecycle.yaml
```

This example calls the public Swagger Petstore, creates a pet and changes its status. It needs internet access and depends on the demo service being available. It doesn't require an API bearer token and doesn't clean up the pet afterwards.

For CLI help:

```sh
uv run restless --help
uv run restless run --help
```

## Write a scenario

Here's the [included Petstore scenario](examples/petstore/pet-lifecycle.yaml):

```yaml
name: pet lifecycle

apis:
    petstore:
        url: https://petstore3.swagger.io/api/v3
        openapi: https://petstore3.swagger.io/api/v3/openapi.json

do: Add a new pet with status "available", then fetch it by its id, then change its status to "sold" and fetch it again.

expect:
    - adding the pet returns the pet with an id
    - fetching it by id returns the same name and status "available"
    - after the update, fetching it again shows status "sold"
```

`name`, `apis`, `do` and `expect` are required. Use `do` to describe the actions and `expect` for the things that must be true. Unknown keys are rejected, so a typo won't silently drop an expectation.

Each API needs a `url` and an `openapi` source. Local spec paths are relative to the scenario file, not the terminal's working directory. For example, `openapi: ./openapi.yaml` reads a spec next to the scenario. Add more entries under `apis` when a flow crosses services, and refer to those API names in your instructions.

The optional `as` field is accepted but doesn't switch identities yet.

### Authentication

For a single token shared by all APIs, set `RESTLESS_TOKEN` in your shell before running the command:

```sh
export RESTLESS_TOKEN="your-test-api-token"
uv run restless run path/to/scenario.yaml
```

```powershell
$env:RESTLESS_TOKEN = "your-test-api-token"
uv run restless run path/to/scenario.yaml
```

The CLI also accepts `--token` / `-t`. A CLI or shell token overrides every API's optional `token` value in the YAML. Set `RESTLESS_TOKEN` in the shell, not in `.env`; the CLI reads it separately from application settings.

For APIs that need different tokens, use the per-API `token` field in a local scenario file and keep that file out of Git. Tokens are supplied upfront; the runner doesn't acquire or refresh them. Remote specs use the corresponding API's bearer token too, so check the spec URL before running.

### Test data and model access

Run against test environments with disposable data. The executor can create, update and delete resources, and cleanup isn't automatic. Scenario text, endpoint details and API responses go to the configured OpenAI models. Bearer headers are added by the HTTP client rather than handed to the executor, but request and response content can still contain sensitive data.

## Results and exit codes

The terminal report lists the scenario name, verdict, HTTP-call count and executor-turn count, followed by numbered requests and the reasoning. When an expectation fails, it also includes the first failure. The banner goes to stderr; the report goes to stdout.

| Exit code | Meaning                                                                                        |
| --------- | ---------------------------------------------------------------------------------------------- |
| `0`       | Pass: the judge found every expectation met.                                                   |
| `1`       | Fail: an expectation wasn't met, or a hard check found a timeout or HTTP 5xx response.         |
| `2`       | Usage error: an unknown command or option, or a scenario file that doesn't exist. Nothing ran. |
| `3`       | Setup/runtime error: settings, scenario loading, specs or model calls prevented a verdict.     |
| `4`       | Inconclusive: evidence is missing, or the executor reached a budget before finishing.          |

To save the terminal report:

```sh
uv run restless run examples/petstore/pet-lifecycle.yaml > report.txt
```

This saves the text summary (UTF-8), not the full HTTP call log. Markdown report export isn't implemented yet.

## Configuration

Settings load from environment variables and `.env` in the current working directory. Environment variables take precedence. The [example file](.env.example) lists the application settings; `.env` is ignored by Git.

| Variable                    | Default  | Purpose                                                           |
| --------------------------- | -------- | ----------------------------------------------------------------- |
| `OPENAI_API_KEY`            | Required | Key for model calls.                                              |
| `EXECUTOR_MODEL`            | Required | Model that chooses and executes API requests.                     |
| `JUDGE_MODEL`               | Required | Model that reviews the call log.                                  |
| `EXECUTOR_REASONING_EFFORT` | Unset    | Optional effort for the executor model.                           |
| `JUDGE_REASONING_EFFORT`    | Unset    | Optional effort for the judge model.                              |
| `HTTP_TIMEOUT_SECONDS`      | `30`     | HTTP client timeout, in seconds.                                  |
| `MAX_HTTP_CALLS`            | `20`     | HTTP-call budget per scenario.                                    |
| `MAX_TURNS`                 | `40`     | Executor-turn budget, including turns used for lookups and waits. |
| `LOG_LEVEL`                 | `INFO`   | `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL`.                |

The HTTP-call budget is checked before each request is sent, so a run never sends more than `MAX_HTTP_CALLS` requests; a run that needs more ends as inconclusive. These are execution limits, not a spending cap.

## Development

`uv sync --locked` installs the development tools too. Unit tests use mocks and don't need a live API or paid model calls.

| Task                 | Command                             |
| -------------------- | ----------------------------------- |
| Unit tests           | `uv run pytest`                     |
| Lint                 | `uv run ruff check .`               |
| Check formatting     | `uv run ruff format --check .`      |
| Format               | `uv run ruff format .`              |
| Type check           | `uv run mypy`                       |
| Install commit hooks | `uv run pre-commit install`         |
| Run all commit hooks | `uv run pre-commit run --all-files` |

The `integration` and `llm` pytest markers are configured and excluded by default. Their directories are placeholders for now. Once tests are added, run them with `uv run pytest -m integration` or `uv run pytest -m llm`.

| Path                     | What's there                                  |
| ------------------------ | --------------------------------------------- |
| `src/restless/cli.py`    | Commands and terminal output.                 |
| `src/restless/runner.py` | Specs, clients, models and graph setup.       |
| `src/restless/actor/`    | Executor, judge and prompts.                  |
| `src/restless/graph/`    | Execution loop, budgets and hard checks.      |
| `src/restless/openapi/`  | Spec loading and operation lookup.            |
| `src/restless/tools/`    | HTTP requests, endpoint details and waits.    |
| `src/restless/model/`    | Scenario, API configuration and report types. |
| `tests/unit/`            | Unit tests.                                   |
| `examples/petstore/`     | A scenario to try.                            |

## Alpha limits and plans

The current entry point is the CLI. There's no HTTP service or MCP server yet. Responses aren't validated against OpenAPI schemas, judge evidence isn't independently verified in code, and runs aren't persisted for replay. Waiting is a simple pause of up to 60 seconds per call; condition-based polling is still planned.

The [v1 design notes](design.md) describe the intended direction, including schema validation, stronger evidence checks, Markdown reports, an MCP wrapper and scenario generation. They're planning notes, so some names and examples differ from the current implementation. Use this README for setup and scenario syntax.

## Contributing

Bug reports and small, focused fixes are welcome. For larger changes, open an issue first so we can discuss the scope while the alpha is still changing.

Include the command, a minimal scenario, the expected result and what happened. Remove tokens and private request or response data before sharing logs. For code changes, run the unit tests, lint, formatting check and type check above.

## License

This repository doesn't include a license file yet.
