from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from restless.tools.http_tool import HttpCall


class JudgeReport(BaseModel):
    """The judge's structured answer. Field order is the order the model writes them: reasoning before the verdict.

    No field has a default. OpenAI's strict structured output rejects a schema with "default" in it,
    and a default would also hide a judge that answered badly behind a plausible-looking value.
    """

    reasoning: str = Field(description="One short paragraph per expectation: its outcome, the call indexes and the values checked.")
    first_failure: str = Field(description="The first expectation not met, with call index, field, expected and actual value. Empty string if none was contradicted.")
    verdict: Literal["pass", "fail", "inconclusive"] = Field(description="pass: every expectation met. fail: any not met. inconclusive: none contradicted but some without evidence.")


@dataclass
class Report:
    result: JudgeReport
    call_log: list[HttpCall]
    hard_failures: list[str]
    steps: int
