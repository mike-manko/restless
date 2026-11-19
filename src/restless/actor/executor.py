from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.tools import BaseTool

from restless.actor.prompts import EXECUTOR_SYSTEM_PROMPT


class Executor:
    """Executor class responsible for managing the execution of a chat model with associated tools."""

    def __init__(self, model: BaseChatModel, tools: Sequence[BaseTool]):
        self.model = model.bind_tools(tools, parallel_tool_calls=False)

    async def step(self, previous_messages: Sequence[BaseMessage]) -> AIMessage:
        """Execute the api calls for the provided scenario"""

        messages = [SystemMessage(content=EXECUTOR_SYSTEM_PROMPT), *previous_messages]
        result = await self.model.ainvoke(messages)
        return result
