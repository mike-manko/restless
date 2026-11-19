from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from restless.tools.http_tool import HttpCall


class JudgeReport(BaseModel):
    reasoning: str = Field("Summary reasoning failed.", description="Provides the reasoning behind the test result")
    first_failure: str = Field("", description="Provides the first failure encountered during the test")
    verdict: Literal["pass", "fail", "inconclusive"] = Field("inconclusive", description="The verdict of the test. Pass indicates success, fail indicates failure, inconclusive indicates uncertainty.")


@dataclass
class Report:
    result: JudgeReport
    call_log: list[HttpCall]
    hard_failures: list[str]
    steps: int
