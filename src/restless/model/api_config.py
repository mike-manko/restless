from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, SecretStr, StringConstraints

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ApiConfig(BaseModel):
    """ApiConfig class describing one API under test: where it lives, where its spec is and which token to send."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    url: NonEmptyStr = Field(description="Base URL. Spec paths are joined to it, e.g. https://localhost:5001/api.")
    openapi: NonEmptyStr = Field(description="URL of the OpenAPI document, or a file path relative to the scenario file.")
    # SecretStr keeps the token out of repr() and logs. The --token option overrides it for every API:
    token: SecretStr | None = Field(default=None, description="Bearer token sent to this API. Leave out for an API without auth.")
