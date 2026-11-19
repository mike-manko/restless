from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from restless.actor.prompts import JUDGE_SYSTEM_PROMPT, build_judge_prompt
from restless.model.report import JudgeReport
from restless.tools.http_tool import HttpCall


class Judge:
    """Judge class responsible for evaluating responses from the api and their correctness."""

    def __init__(self, model: BaseChatModel):
        self.model = model.with_structured_output(JudgeReport)

    async def report(self, scenario: str, call_log: list[HttpCall]) -> JudgeReport:
        """Evaluate the result of the test based on the provided call log."""

        messages = [SystemMessage(content=JUDGE_SYSTEM_PROMPT), HumanMessage(content=build_judge_prompt(scenario, call_log))]
        result = await self.model.ainvoke(messages)

        return JudgeReport.model_validate(result)
