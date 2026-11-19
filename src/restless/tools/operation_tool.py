import json

from langchain_core.tools import BaseTool, StructuredTool, ToolException
from pydantic import BaseModel, Field

from restless.openapi.catalog import ApiCatalog, UnknownOperationError


class GetOperationArgs(BaseModel):
    """What the model fills in. The field descriptions are sent to the model with the tool."""

    api: str = Field(description="Name of the API, as in the operation list.")
    operation: str = Field(description='The operation as listed, e.g. "POST /orders/{id}", or its operationId.')


def create_operation_tool(catalog: ApiCatalog) -> BaseTool:
    """Build the get_operation tool on top of this suite's catalog.

    Plain text in, plain text out, and no artifact: a lookup isn't API behaviour, so
    call_tools keeps it out of the call log and the judge never sees it.
    """

    def get_operation(api: str, operation: str) -> str:
        # An unknown name becomes a message the model can read and correct, like an unknown API in http_request:
        try:
            found = catalog.find(api, operation)
        except UnknownOperationError as exc:
            raise ToolException(str(exc)) from exc

        details = catalog.details(found)

        return json.dumps(details, indent=2, ensure_ascii=False)

    return StructuredTool.from_function(
        func=get_operation,
        name="get_operation",
        description=(
            "Get one operation's parameters, request body schema and response schemas from the API's OpenAPI spec. "
            "Call it before your first request to an operation, so you send the fields the API actually expects."
        ),
        args_schema=GetOperationArgs,
        handle_tool_error=True,
        handle_validation_error=lambda exc: f"Invalid arguments: {exc}",
    )
