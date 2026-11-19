import logging
from contextlib import AsyncExitStack
from pathlib import Path

import httpx
from azure.identity import DefaultAzureCredential
from langchain_openai import ChatOpenAI

from restless.actor.executor import Executor
from restless.actor.judge import Judge
from restless.config import Settings
from restless.graph.builder import build_graph
from restless.graph.context import RunContext
from restless.model.api_config import ApiConfig, SuiteConfig
from restless.model.report import Report
from restless.model.scenario import Scenario
from restless.openapi.catalog import ApiCatalog
from restless.openapi.loader import load_specs
from restless.tools.api_clients import create_api_client, create_entra_auth, new_traceparent
from restless.tools.http_tool import create_http_tool
from restless.tools.operation_tool import create_operation_tool

logger = logging.getLogger(__name__)


# Everything that happens before and around the graph: config, specs, clients, tools, models
async def run_scenario(scenario: Scenario, config: SuiteConfig, *, config_dir: Path, settings: Settings) -> Report:
    """Run one scenario end to end and return its report.

    Order matters: everything that can fail without an LLM (unknown API names, missing
    specs) happens first, so a broken setup costs nothing.
    """

    apis = _select_apis(config, scenario)
    auths = _create_auths(apis)

    # Once per suite in the future (shared by every scenario); once per run for now:
    specs = await load_specs(apis, auths, base_dir=config_dir, timeout_seconds=settings.http_timeout_seconds)
    catalog = ApiCatalog(specs)
    # Field names, not the aliases model= and api_key=: the pydantic mypy plugin only knows the field names:
    model = ChatOpenAI(model_name=settings.openai_model, openai_api_key=settings.openai_api_key)

    # Once per run: one trace ID shared by every call, so the run can be found in Application Insights:
    traceparent = new_traceparent()
    logger.info("Running %r with traceparent %s", scenario.name, traceparent)

    async with AsyncExitStack() as stack:
        clients: dict[str, httpx.AsyncClient] = {}
        for name, api in apis.items():
            client = create_api_client(api.url, traceparent=traceparent, timeout_seconds=settings.http_timeout_seconds, auth=auths[name])
            clients[name] = await stack.enter_async_context(client)

        tools = [create_http_tool(clients), create_operation_tool(catalog)]
        context = RunContext(
            settings=settings,
            tools={tool.name: tool for tool in tools},
            catalog=catalog,
            executor=Executor(model, tools),
            judge=Judge(model),
            max_http_calls=settings.max_http_calls,
            max_turns=settings.max_turns,
        )

        # Each executor turn is two graph steps (executor, tools). LangGraph's default limit of 25 steps would stop a long run before our own budget does:
        recursion_limit = settings.max_turns * 2 + 10
        result = await build_graph().ainvoke({"scenario": scenario}, context=context, config={"recursion_limit": recursion_limit})

    report: Report = result["report"]

    return report


def _select_apis(config: SuiteConfig, scenario: Scenario) -> dict[str, ApiConfig]:
    """The APIs this scenario may call: the ones it names, or all configured ones."""

    if scenario.apis is None:
        return dict(config.apis)

    unknown = [name for name in scenario.apis if name not in config.apis]
    if unknown:
        raise ValueError(f"Scenario {scenario.name!r} uses APIs that aren't configured: {', '.join(unknown)}. Configured: {', '.join(config.apis)}.")

    return {name: config.apis[name] for name in scenario.apis}


def _create_auths(apis: dict[str, ApiConfig]) -> dict[str, httpx.Auth | None]:
    """One bearer auth per API with a scope. All of them share one credential, so sign-in happens once."""

    # DefaultAzureCredential tries environment variables, managed identity, then `az login` and others, in that order:
    credential = DefaultAzureCredential() if any(api.scope for api in apis.values()) else None

    auths: dict[str, httpx.Auth | None] = {}
    for name, api in apis.items():
        auths[name] = create_entra_auth(credential, api.scope) if credential and api.scope else None

    return auths
