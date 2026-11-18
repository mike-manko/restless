# restless

Typed Python 3.13 service: a Typer CLI, a FastAPI HTTP API, an HTTPX outbound client and an LLM stack (LangGraph, OpenAI), with Pydantic settings.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync                    # create .venv and install locked packages
cp .env.example .env       # then fill in OPENAI_API_KEY and OPENAI_MODEL
uv run pre-commit install  # run checks on every commit
```

`.env` is git-ignored; `.env.example` lists every setting and is committed.

## Commands

| Task | Command |
| --- | --- |
| Run the app | `uv run restless` |
| Unit tests | `uv run pytest` |
| Integration tests | `uv run pytest -m integration` |
| LLM tests (paid, need an API key) | `uv run pytest -m llm` |
| Lint | `uv run ruff check .` |
| Format | `uv run ruff format .` |
| Type check | `uv run mypy` |
| All pre-commit checks | `uv run pre-commit run --all-files` |
| Add a package | `uv add <name>` (dev tool: `uv add --dev <name>`) |

## Layout

| Path | Purpose |
| --- | --- |
| `src/restless/` | The application package |
| `src/restless/config.py` | `Settings`, bound from environment variables and `.env` |
| `src/restless/logging_setup.py` | `configure_logging()` for entry points |
| `src/restless/http/generated/` | OpenAPI-generated clients (excluded from Ruff, relaxed in mypy) |
| `tests/unit`, `tests/integration`, `tests/llm` | Tests by cost; only unit tests run by default |
| `tests/fixtures/` | Test data files |
| `scripts/` | Maintenance scripts: type-checked, not shipped |
