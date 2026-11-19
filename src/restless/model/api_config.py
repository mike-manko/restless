from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field

from restless.model.scenario import NonEmptyStr


class ApiConfig(BaseModel):
    """ApiConfig class describing one API under test: where it lives, where its spec is and how to get a token for it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    url: NonEmptyStr = Field(description="Base URL. Spec paths are joined to it, e.g. https://localhost:5001/api.")
    openapi: NonEmptyStr = Field(description="URL of the OpenAPI document, or a file path relative to the config file.")
    scope: NonEmptyStr | None = Field(default=None, description="Entra ID scope, e.g. api://orders/.default. Leave out for an API without auth.")


class SuiteConfig(BaseModel):
    """SuiteConfig class holding every API a suite can call. Loaded once and shared by all scenarios."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    apis: dict[str, ApiConfig] = Field(min_length=1, description="API name -> config. Tools and scenarios refer to APIs by these names.")

    @classmethod
    def from_file(cls, path: Path) -> Self:
        """Load and validate the API config from a YAML file."""

        data = yaml.safe_load(path.read_text(encoding="utf-8"))

        return cls.model_validate(data)
