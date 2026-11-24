import logging
from contextlib import AsyncExitStack
from pathlib import Path
from typing import assert_never

import httpx
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from restless.actor.executor import Executor
from restless.actor.judge import Judge
from restless.graph.builder import build_graph
from restless.graph.context import RunContext
from restless.model.api_config import ApiConfig
from restless.model.report import Report
from restless.model.scenario import Scenario
from restless.openapi.catalog import ApiCatalog
from restless.openapi.loader import load_specs
from restless.settings import AnthropicModelSettings, AzureOpenAIModelSettings, ModelSettings, OpenAIModelSettings, Settings
from restless.tools.api_clients import BearerTokenAuth, create_api_client, new_traceparent
from restless.tools.http_tool import create_http_tool
from restless.tools.operation_tool import create_operation_tool
from restless.tools.wait_tool import create_wait_tool

logger = logging.getLogger(__name__)


# Everything that happens before and around the graph: config, specs, clients, tools, models
async def run_scenario(scenario: Scenario, *, base_dir: Path, settings: Settings, token: str | None = None) -> Report:
    """Run one scenario end to end and return its report.

    Order matters: everything that can fail without an LLM (missing specs) happens first,
    so a broken setup costs nothing. base_dir is where relative spec file paths start: the scenario file's folder.
    """

    apis = dict(scenario.apis)
    auths = _create_auths(apis, token)

    specs = await load_specs(apis, auths, base_dir=base_dir, timeout_seconds=settings.http_timeout_seconds)
    catalog = ApiCatalog(specs)
    executor_model = _create_model(settings.executor)
    judge_model = _create_model(settings.judge)

    # Once per run: one trace ID shared by every call, so the run can be found in Application Insights:
    traceparent = new_traceparent()
    logger.info("Running %r with traceparent %s", scenario.name, traceparent)

    async with AsyncExitStack() as stack:
        clients: dict[str, httpx.AsyncClient] = {}

        for name, api in apis.items():
            client = create_api_client(api.url, traceparent=traceparent, timeout_seconds=settings.http_timeout_seconds, auth=auths[name])
            clients[name] = await stack.enter_async_context(client)

        tools = [create_http_tool(clients), create_operation_tool(catalog), create_wait_tool()]
        context = RunContext(
            tools={tool.name: tool for tool in tools},
            catalog=catalog,
            executor=Executor(executor_model, tools),
            judge=Judge(judge_model),
            max_http_calls=settings.max_http_calls,
            max_turns=scenario.max_turns if scenario.max_turns is not None else settings.default_max_turns,
        )

        # Each executor turn is two graph steps (executor, tools), so this limit sits just above the turn cap, which ends every normal run first.
        # It stops a routing mistake early: LangGraph's own default would let one loop for thousands of steps:
        recursion_limit = context.max_turns * 2 + 10
        result = await build_graph().ainvoke({"scenario": scenario}, context=context, config={"recursion_limit": recursion_limit})

    report: Report = result["report"]

    return report


def _create_model(model_settings: ModelSettings) -> BaseChatModel:
    """One chat model for the provider in the settings. reasoning_effort=None leaves the parameter out of the request, which models without reasoning require."""

    # Field names, not the aliases model= and api_key=: the pydantic mypy plugin only knows the field names, so it checks every keyword.
    # A dict passed to model_validate isn't checked at all, and LangChain moves an unknown key into model_kwargs with only a warning:
    if isinstance(model_settings, OpenAIModelSettings):
        return ChatOpenAI(model_name=model_settings.model, openai_api_key=model_settings.api_key, reasoning_effort=model_settings.reasoning_effort)

    if isinstance(model_settings, AzureOpenAIModelSettings):
        return AzureChatOpenAI(
            azure_endpoint=model_settings.endpoint,
            deployment_name=model_settings.deployment,
            openai_api_version=model_settings.api_version,
            openai_api_key=model_settings.api_key,
            reasoning_effort=model_settings.reasoning_effort,
        )

    if isinstance(model_settings, AnthropicModelSettings):
        return ChatAnthropic(model=model_settings.model, anthropic_api_key=model_settings.api_key, reasoning_effort=model_settings.reasoning_effort)

    # Every provider is handled above, so mypy types model_settings as Never here. A new provider in ModelSettings makes this line a mypy error until it gets a branch:
    assert_never(model_settings)


def _create_auths(apis: dict[str, ApiConfig], token: str | None) -> dict[str, httpx.Auth | None]:
    """One bearer auth per API: the token passed in for every API if there is one, otherwise each API's configured token."""

    auths: dict[str, httpx.Auth | None] = {}
    for name, api in apis.items():
        # The token from the command line wins over the scenario file. No token at all means the API is called without auth:
        api_token = token or (api.token.get_secret_value() if api.token else None)
        auths[name] = BearerTokenAuth(api_token) if api_token else None

    return auths
