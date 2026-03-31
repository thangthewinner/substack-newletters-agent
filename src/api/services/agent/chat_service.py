from collections.abc import AsyncGenerator
from typing import Protocol

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from sqlalchemy.engine import Engine

from src.api.models.api_models import ChatMessage
from src.api.services.agent.llm_factory import create_agent_llm
from src.api.services.agent.prompts.system_prompt import SYSTEM_PROMPT
from src.api.services.agent.tools import create_tools
from src.config import settings
from src.infrastructure.qdrant.qdrant_vectorstore import AsyncQdrantVectorStore
from src.utils.logger_util import setup_logging

logger = setup_logging()
TOOL_CALL_STEP_TAG = ["seq:step:1"]


class AgentRunnable(Protocol):
    """Protocol for chat agent invocation and streaming interfaces."""

    async def ainvoke(self, input_data: dict) -> dict: ...

    def astream_events(
        self, input_data: dict, version: str
    ) -> AsyncGenerator[dict, None]: ...


def build_lc_messages(messages: list[ChatMessage]) -> list[HumanMessage | AIMessage]:
    """Convert ChatMessage list to LangChain message objects."""
    result: list[HumanMessage | AIMessage] = []
    for m in messages:
        if m.role == "user":
            result.append(HumanMessage(content=m.content))
        elif m.role == "assistant":
            result.append(AIMessage(content=m.content))
    return result


def create_agent(
    vectorstore: AsyncQdrantVectorStore,
    db_engine: Engine,
    model: str | None = None,
) -> AgentRunnable:
    """Build and return a LangGraph ReAct agent."""
    llm = create_agent_llm(model)
    tools = create_tools(vectorstore, db_engine)
    return create_react_agent(
        llm,
        tools=tools,
        prompt=SystemMessage(content=SYSTEM_PROMPT),
    )


async def run_chat(
    agent: AgentRunnable,
    messages: list[ChatMessage],
) -> str:
    """Run agent non-streaming. Returns final reply string."""
    lc_messages = build_lc_messages(messages)
    result = await agent.ainvoke({"messages": lc_messages})
    return result["messages"][-1].content


async def run_chat_stream(
    agent: AgentRunnable,
    messages: list[ChatMessage],
) -> AsyncGenerator[str, None]:
    """Run agent with streaming. Yields text chunks."""
    lc_messages = build_lc_messages(messages)
    async for event in agent.astream_events(
        {"messages": lc_messages}, version=settings.agent.stream_version
    ):
        if (
            event["event"] == "on_chat_model_stream"
            and event.get("tags", []) != TOOL_CALL_STEP_TAG
        ):
            chunk = event["data"]["chunk"].content
            if chunk:
                yield chunk
