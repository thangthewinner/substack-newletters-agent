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
from src.api.services.search_service import query_unique_titles
from src.config import settings

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@router.post("/unique-titles", response_model=UniqueTitleResponse)
async def search_unique(request: Request, params: UniqueTitleRequest):
    """
    Returns unique article titles based on a query and optional filters.

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
    """
    Chat endpoint using LangGraph ReAct agent.

    Args:
        request: FastAPI request object.
        body: ChatRequest with conversation history.

    Returns:
        ChatResponse: Agent's final answer in Markdown.

    """
    agent = request.app.state.agent
    reply = await run_chat(agent, body.messages, session_id=body.session_id)
    return ChatResponse(reply=reply)


@router.post("/chat/stream")
@limiter.limit(settings.agent.rate_limit)
async def chat_stream(request: Request, body: ChatRequest):
    """
    Streaming chat endpoint using LangGraph ReAct agent.

    Args:
        request: FastAPI request object.
        body: ChatRequest with conversation history.

    Returns:
        StreamingResponse: Yields text chunks as text/event-stream.

    """
    agent = request.app.state.agent

    async def generator():
        async for chunk in run_chat_stream(
            agent,
            body.messages,
            session_id=body.session_id,
        ):
            yield chunk

    return StreamingResponse(generator(), media_type="text/event-stream")
