def main() -> None:
    """Entry point for the `restless` command (see [project.scripts] in pyproject.toml)."""

    # Imported here, so `import restless` stays cheap and doesn't load LangChain, httpx and Azure:
    from restless.cli import app

    app()
