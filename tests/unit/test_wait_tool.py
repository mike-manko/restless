from langchain_core.messages import ToolMessage
from langchain_core.messages.tool import tool_call

from restless.tools.wait_tool import create_wait_tool


async def test_wait_returns_text_for_the_model() -> None:
    message = await create_wait_tool().ainvoke(tool_call(name="wait", args={"seconds": 1}, id="call-1"))

    assert isinstance(message, ToolMessage)
    assert message.content == "Waited 1 seconds."


async def test_wait_longer_than_the_limit_is_an_error_the_model_can_read() -> None:
    message = await create_wait_tool().ainvoke(tool_call(name="wait", args={"seconds": 3600}, id="call-1"))

    assert isinstance(message, ToolMessage)
    assert message.status == "error"
    assert "Invalid arguments" in str(message.content)
