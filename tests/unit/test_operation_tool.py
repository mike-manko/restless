import json
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.messages.tool import tool_call

from restless.openapi.catalog import ApiCatalog
from restless.tools.operation_tool import create_operation_tool

SPEC: dict[str, Any] = {
    "paths": {
        "/orders/{id}": {
            "get": {
                "operationId": "GetOrder",
                "summary": "Get one order",
                "parameters": [{"name": "id", "in": "path", "required": True, "schema": {"type": "integer"}}],
                "responses": {"200": {"description": "OK"}},
            }
        }
    }
}


def look_up(operation: str) -> ToolMessage:
    """Invoke get_operation with a tool call, which returns a ToolMessage."""

    tool = create_operation_tool(ApiCatalog({"orders": SPEC}))
    message = tool.invoke(tool_call(name="get_operation", args={"api": "orders", "operation": operation}, id="call-1"))
    assert isinstance(message, ToolMessage)

    return message


def test_lookup_returns_the_operation_as_json_and_no_artifact() -> None:
    message = look_up("GetOrder")

    details = json.loads(message.text)
    assert details["operation"] == "GET /orders/{id}"
    assert details["parameters"][0]["name"] == "id"
    # No artifact: a lookup isn't API behaviour, so it stays out of the call log:
    assert message.artifact is None


def test_unknown_operation_is_an_error_the_model_can_correct() -> None:
    message = look_up("GET /order/{id}")

    assert message.status == "error"
    assert "Did you mean: GET /orders/{id}?" in message.text
