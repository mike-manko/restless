from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.ai import UsageMetadata

from restless.actor.prompts import JUDGE_SYSTEM_PROMPT, build_judge_prompt
from restless.model.report import JudgeReport
from restless.tools.http_tool import HttpCall


class Judge:
    """Judge class responsible for evaluating responses from the api and their correctness."""

    def __init__(self, model: BaseChatModel):
        # include_raw keeps the model's reply next to the parsed report, so the reply's token usage isn't lost:
        self.model = model.with_structured_output(JudgeReport, include_raw=True)

    async def report(self, scenario: str, call_log: list[HttpCall]) -> tuple[JudgeReport, UsageMetadata | None]:
        """Evaluate the result of the test based on the provided call log. Returns the report and the tokens this call used, None if the provider reports none."""

        messages = [SystemMessage(content=JUDGE_SYSTEM_PROMPT), HumanMessage(content=build_judge_prompt(scenario, call_log))]
        result = await self.model.ainvoke(messages)

        # With include_raw the answer is a dict: the raw reply, the parsed report and the error if parsing failed:
        if not isinstance(result, dict):
            raise TypeError("Expected the raw reply and the parsed report")

        # include_raw hands back a parsing error instead of raising it. Raise it, so a judge that answered badly is a setup error, not a verdict:
        if result["parsing_error"] is not None:
            raise result["parsing_error"]

        report = JudgeReport.model_validate(result["parsed"])
        usage: UsageMetadata | None = result["raw"].usage_metadata

        return report, usage
