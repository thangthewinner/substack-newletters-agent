"""Chat Service."""

from collections.abc import AsyncGenerator
from typing import Any, Protocol

from langchain.agents import create_agent as create_langchain_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from sqlalchemy.engine import Engine

from src.api.models.api_models import ChatMessage
from src.api.services.agent.llm_factory import create_agent_llm
from src.api.services.agent.prompts.system_prompt import SYSTEM_PROMPT
from src.api.services.agent.tools.search_tools import create_search_tools
from src.api.services.agent.tools.sql_tools import create_sql_tools
from src.config import settings
from src.infrastructure.qdrant.qdrant_vectorstore import AsyncQdrantVectorStore
from src.utils.logger_util import setup_logging

logger = setup_logging()
TOOL_CALL_STEP_TAG = ["seq:step:1"]


class AgentRunnable(Protocol):
    """Protocol for chat agent invocation and streaming interfaces."""

    async def ainvoke(self, input_data: dict, **kwargs: Any) -> dict:
        """Invoke the agent asynchronously."""

    def astream_events(
        self, input_data: dict, version: str, **kwargs: Any
    ) -> AsyncGenerator[dict, None]:
        """Stream events from the agent asynchronously."""


def build_lc_messages(
    messages: list[ChatMessage], max_history_messages: int
) -> list[HumanMessage | AIMessage]:
    """Convert ChatMessage list to LangChain message objects with history trimming."""
    if max_history_messages > 0 and len(messages) > max_history_messages:
        messages = messages[-max_history_messages:]

    result: list[HumanMessage | AIMessage] = []
    for message in messages:
        if message.role == "user":
            result.append(HumanMessage(content=message.content))
        elif message.role == "assistant":
            result.append(AIMessage(content=message.content))
    return result


def _build_input_payload(
    messages: list[ChatMessage],
    session_id: str | None,
) -> dict[str, list[HumanMessage | AIMessage]]:
    """Build agent payload, using only latest user message for session mode."""
    if session_id:
        for message in reversed(messages):
            if message.role == "user":
                return {"messages": [HumanMessage(content=message.content)]}
        return {"messages": []}

    return {
        "messages": build_lc_messages(
            messages=messages,
            max_history_messages=settings.agent.max_history_messages,
        )
    }


def create_agent(
    vectorstore: AsyncQdrantVectorStore,
    db_engine: Engine,
    model: str | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> AgentRunnable:
    """Build and return a LangGraph ReAct agent."""
    llm = create_agent_llm(model)
    search_tools = create_search_tools(vectorstore)
    sql_tools = create_sql_tools(db_engine)
    tools = [*search_tools, *sql_tools]
    if checkpointer is None:
        checkpointer = MemorySaver()
    return create_langchain_agent(
        llm,
        tools=tools,
        system_prompt=SystemMessage(content=SYSTEM_PROMPT),
        checkpointer=checkpointer,
    )


async def run_chat(
    agent: AgentRunnable,
    messages: list[ChatMessage],
    session_id: str | None = None,
) -> str:
    """Run agent non-streaming. Returns final reply string."""
    payload = _build_input_payload(messages, session_id)
    if session_id:
        result = await agent.ainvoke(
            payload,
            config={"configurable": {"thread_id": session_id}},
        )
    else:
        result = await agent.ainvoke(payload)
    return result["messages"][-1].content


async def run_chat_stream(
    agent: AgentRunnable,
    messages: list[ChatMessage],
    session_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """Run agent with streaming. Yields text chunks."""
    payload = _build_input_payload(messages, session_id)
    if session_id:
        stream = agent.astream_events(
            payload,
            version=settings.agent.stream_version,
            config={"configurable": {"thread_id": session_id}},
        )
    else:
        stream = agent.astream_events(
            payload,
            version=settings.agent.stream_version,
        )

    first_event_logged = False
    yielded_chunks = 0
    fallback_emitted = False
    try:
        async for event in stream:
            if not first_event_logged:
                first_event_logged = True

            if event["event"] == "on_tool_start":
                tool_name = event.get("name", "tool")
                yield f"__TOOL_START__:{tool_name}\n"

            if (
                event["event"] == "on_chat_model_stream"
                and event.get("tags", []) != TOOL_CALL_STEP_TAG
            ):
                chunk = event["data"]["chunk"].content
                if chunk:
                    yielded_chunks += 1
                    yield chunk

            if event["event"] == "on_chain_end":
                output_data = event.get("data", {}).get("output")
                fallback_reply: str | None = None
                if isinstance(output_data, dict):
                    messages_data = output_data.get("messages")
                    if isinstance(messages_data, list):
                        if yielded_chunks == 0 and not fallback_emitted:
                            for msg in reversed(messages_data):
                                if isinstance(msg, AIMessage):
                                    content = msg.content
                                    if isinstance(content, str) and content.strip():
                                        fallback_reply = content
                                        break
                if fallback_reply is not None:
                    fallback_emitted = True
                    yield fallback_reply
    except Exception as exc:
        logger.exception(f"Error while streaming chat response: {exc}")
        raise
