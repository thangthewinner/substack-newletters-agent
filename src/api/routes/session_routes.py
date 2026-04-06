"""Session Routes."""
from fastapi import APIRouter, HTTPException, Request
# Empty import line removed

from src.infrastructure.supabase.session_repository import (
    create_session,
    delete_session,
    get_session,
    list_sessions,
    update_session_name,
)
from src.models.session_models import (
    SessionCreate,
    SessionDetailResponse,
    SessionInfo,
    SessionUpdate,
)

router = APIRouter()


@router.post("/sessions", response_model=SessionInfo)
async def create_new_session(request: Request, body: SessionCreate):
    """Create a new chat session."""
    engine = request.app.state.db_engine
    return create_session(engine, body)


@router.get("/sessions", response_model=list[SessionInfo])
async def list_all_sessions(request: Request, limit: int = 50):
    """List all sessions sorted by last_message_at DESC."""
    engine = request.app.state.db_engine
    return list_sessions(engine, limit)


@router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
async def get_session_detail(request: Request, session_id: str):
    """Get session detail including messages."""
    engine = request.app.state.db_engine
    session = get_session(engine, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.patch("/sessions/{session_id}")
async def update_session(request: Request, session_id: str, body: SessionUpdate):
    """Update session name."""
    engine = request.app.state.db_engine
    update_session_name(engine, session_id, body.name)
    return {"status": "ok"}


@router.delete("/sessions/{session_id}")
async def delete_session_route(request: Request, session_id: str):
    """Delete session metadata and LangGraph checkpoints."""
    engine = request.app.state.db_engine
    delete_session(engine, session_id)

    # Delete all LangGraph checkpoint data for this thread.
    checkpointer = request.app.state.checkpointer
    if hasattr(checkpointer, "adelete_thread"):
        await checkpointer.adelete_thread(session_id)
    return {"status": "ok"}
