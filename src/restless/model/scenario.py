from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field

from restless.model.api_config import ApiConfig, NonEmptyStr


class Scenario(BaseModel):
    """Scenario class describing one complete test in one file: the APIs it calls, what to do and what must be true afterwards."""

    # Unknown keys are an error, so a misspelled "expect:" fails instead of running a test with no expectations.
    # validate_by_name lets code and tests write as_user=..., while the YAML file uses "as":
    model_config = ConfigDict(extra="forbid", frozen=True, validate_by_name=True, validate_by_alias=True)

    name: NonEmptyStr = Field(description="Short name of the test, shown in the report.")
    as_user: NonEmptyStr | None = Field(default=None, alias="as", description="Identity to run as. Not used yet: every API gets the one configured identity.")
    apis: dict[str, ApiConfig] = Field(min_length=1, description="API name -> where it lives, its spec and its token. Tools refer to APIs by these names.")
    do: NonEmptyStr = Field(description="What the executor should do, in plain language.")
    expect: list[NonEmptyStr] = Field(min_length=1, description="What must be true afterwards, one expectation per item.")
    max_turns: int | None = Field(default=None, gt=0, description="Maximum executor turns for this scenario. Overrides the default if set.")

    @classmethod
    def from_file(cls, path: Path) -> Self:
        """Load and validate a scenario from a YAML file."""

        # safe_load only builds dicts, lists and plain values. yaml.load can construct any Python object a file asks for:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))

        return cls.model_validate(data)

    def to_prompt(self) -> str:
        """The scenario as the executor and the judge see it. Both get the same text."""

        lines: list[str] = [f"name: {self.name}"]
        if self.as_user:
            lines.append(f"as: {self.as_user}")

        lines += [f"do: {self.do}", "expect:"]
        lines += [f"  - {item}" for item in self.expect]

        return "\n".join(lines)
