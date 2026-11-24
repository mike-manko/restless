import os

import pytest

from restless.settings import Settings

# Unit tests never send traces to LangSmith, even when .env or the shell switches tracing on.
# langsmith caches the variable the first time it reads it, so set it here, before any test runs:
os.environ["LANGSMITH_TRACING_V2"] = "false"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every setting and the CLI's RESTLESS_TOKEN from the process environment, so the developer's shell can't leak into unit tests.

    No yield needed: monkeypatch restores the environment itself after each test.
    """

    # Top-level settings (LOG_LEVEL) and nested ones (EXECUTOR__MODEL), which start with the name of their settings field:
    for name in list(os.environ):
        if name.lower().partition("__")[0] in Settings.model_fields:
            monkeypatch.delenv(name)

    monkeypatch.delenv("RESTLESS_TOKEN", raising=False)

    # Rich reads these to force colour on or off, which would change what the report tests see:
    for name in ("FORCE_COLOR", "NO_COLOR"):
        monkeypatch.delenv(name, raising=False)
