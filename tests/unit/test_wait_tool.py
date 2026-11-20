import asyncio

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.messages.tool import tool_call

from restless.tools.wait_tool import create_wait_tool


@pytest.fixture
def slept(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record every sleep instead of sleeping, so no test waits for real. Returns the seconds asked for, in order."""

    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    return slept


async def test_wait_sleeps_the_seconds_asked_for_and_tells_the_model(slept: list[float]) -> None:
    message = await create_wait_tool().ainvoke(tool_call(name="wait", args={"seconds": 5}, id="call-1"))

    assert isinstance(message, ToolMessage)
    assert message.content == "Waited 5 seconds."
    assert slept == [5]


async def test_wait_longer_than_the_limit_is_an_error_the_model_can_read(slept: list[float]) -> None:
    message = await create_wait_tool().ainvoke(tool_call(name="wait", args={"seconds": 3600}, id="call-1"))

    assert isinstance(message, ToolMessage)
    assert message.status == "error"
    assert "Invalid arguments" in str(message.content)
    assert slept == []
