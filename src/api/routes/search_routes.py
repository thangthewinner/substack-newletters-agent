"""Search Routes."""

import asyncio

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.api.models.api_models import (
    ChatRequest,
    ChatResponse,
    UniqueTitleRequest,
    UniqueTitleResponse,
)
from src.api.services.agent.chat_service import run_chat, run_chat_stream
from src.api.services.agent.naming_service import generate_session_name
from src.api.services.search_service import query_unique_titles
from src.config import settings
from src.infrastructure.supabase.session_repository import (
    append_message,
    touch_session,
    update_session_name,
)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


async def _name_and_save(first_message: str, session_id: str, engine) -> None:
    """Generate a session name from the first message and persist it to the DB."""
    name = await generate_session_name(first_message)
    update_session_name(engine, session_id, name)


@router.post("/unique-titles", response_model=UniqueTitleResponse)
async def search_unique(request: Request, params: UniqueTitleRequest):
    """Return unique article titles based on a query and optional filters.

    Deduplicates results by article title.

    Args:
        request: FastAPI request object.
        params: UniqueTitleRequest with search parameters.

    Returns:
        UniqueTitleResponse: List of unique titles.

    """
    vectorstore = request.app.state.vectorstore
    results = await query_unique_titles(
        vectorstore=vectorstore,
        query_text=params.query_text,
        feed_author=params.feed_author,
        feed_name=params.feed_name,
        title_keywords=params.title_keywords,
        limit=params.limit,
    )
    return {"results": results}


@router.post("/chat", response_model=ChatResponse)
@limiter.limit(settings.agent.rate_limit)
async def chat(request: Request, body: ChatRequest):
    """Chat endpoint using LangGraph ReAct agent.

    Args:
        request: FastAPI request object.
        body: ChatRequest with conversation history.

    Returns:
        ChatResponse: Agent's final answer in Markdown.

    """
    agent = request.app.state.agent
    engine = request.app.state.db_engine

    msg_count = append_message(
        engine, body.session_id, "user", body.messages[-1].content
    )
    if msg_count == 1:
        first_msg = body.messages[-1].content
        asyncio.create_task(_name_and_save(first_msg, body.session_id, engine))

    reply = await run_chat(agent, body.messages, session_id=body.session_id)

    append_message(engine, body.session_id, "assistant", reply)
    touch_session(engine, body.session_id)
    return ChatResponse(reply=reply)


@router.post("/chat/stream")
@limiter.limit(settings.agent.rate_limit)
async def chat_stream(request: Request, body: ChatRequest):
    """Streaming chat endpoint using LangGraph ReAct agent.

    Args:
        request: FastAPI request object.
        body: ChatRequest with conversation history.

    Returns:
        StreamingResponse: Yields text chunks as text/event-stream.

    """
    agent = request.app.state.agent
    engine = request.app.state.db_engine

    msg_count = append_message(
        engine, body.session_id, "user", body.messages[-1].content
    )
    if msg_count == 1:
        first_msg = body.messages[-1].content
        asyncio.create_task(_name_and_save(first_msg, body.session_id, engine))

    async def generator():
        full_reply = ""
        async for chunk in run_chat_stream(
            agent,
            body.messages,
            session_id=body.session_id,
        ):
            full_reply += chunk
            yield chunk
        append_message(engine, body.session_id, "assistant", full_reply)
        touch_session(engine, body.session_id)

    return StreamingResponse(generator(), media_type="text/event-stream")
