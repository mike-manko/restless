import asyncio

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

# Long enough for background work to move on, short enough that a confused model can't stall the run:
_MAX_WAIT_SECONDS = 60


class WaitToolArgs(BaseModel):
    """What the model fills in. The field description is sent to the model with the tool."""

    seconds: int = Field(ge=1, le=_MAX_WAIT_SECONDS, description=f"Seconds to wait, 1 to {_MAX_WAIT_SECONDS}.")


def create_wait_tool() -> BaseTool:
    """Build the wait tool: lets the executor give background work (e.g. a calculation workflow) time to progress."""

    # asyncio.sleep instead of time.sleep: the graph runs on an event loop, and a coroutine waits without holding a thread:
    async def wait(seconds: int) -> str:
        await asyncio.sleep(seconds)

        return f"Waited {seconds} seconds."

    return StructuredTool.from_function(
        coroutine=wait,
        name="wait",
        description="Wait before the next request, to give background work time to progress. Use it only when a response shows work is still in progress.",
        args_schema=WaitToolArgs,
        handle_tool_error=True,
        handle_validation_error=lambda exc: f"Invalid arguments: {exc}",
    )
