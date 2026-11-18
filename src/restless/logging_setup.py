"""Logging configuration for entry points (CLI command, API startup, scripts).

Library modules only call `logging.getLogger(__name__)`; the entry point configures output once.
"""

import logging

# Installed libraries that log every request at INFO/DEBUG.
_NOISY_LOGGERS = ("httpx", "httpcore", "openai")


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
