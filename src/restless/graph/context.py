from dataclasses import dataclass

from langchain_core.tools import BaseTool

from restless.actor.executor import Executor
from restless.actor.judge import Judge
from restless.openapi.catalog import ApiCatalog


@dataclass(frozen=True)
class RunContext:
    tools: dict[str, BaseTool]
    catalog: ApiCatalog
    executor: Executor
    judge: Judge
    max_http_calls: int  # the budget: HTTP requests the executor may send
    max_turns: int  # safety net: executor turns in total, lookups and waits included, so a model stuck looking things up still stops
