"""Session Repository."""

import json
from datetime import datetime, timezone

from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.models.session_models import (
    MessageRecord,
    SessionCreate,
    SessionDetailResponse,
    SessionInfo,
)
from src.utils.logger_util import setup_logging

logger = setup_logging()


def create_session(engine: Engine, session: SessionCreate) -> SessionInfo:
    """Create a new chat session in the database.

    Args:
        engine: SQLAlchemy database engine.
        session: Session creation parameters.

    Returns:
        Created session information.

    """
    sql = text(
        """
        INSERT INTO chat_sessions (name, model, messages, first_message_preview, created_at, last_message_at, message_count)
        VALUES (:name, :model, '[]'::jsonb, NULL, NOW(), NOW(), 0)
        RETURNING id, name, model, first_message_preview, created_at, last_message_at, message_count
        """
    )
    with engine.begin() as conn:
        row = conn.execute(
            sql, {"name": session.name, "model": session.model}
        ).fetchone()
    if row is None:
        raise RuntimeError(
            "INSERT INTO chat_sessions returned no row — check DB constraints"
        )
    return SessionInfo(
        id=row.id,
        name=row.name,
        model=row.model,
        first_message_preview=row.first_message_preview,
        created_at=row.created_at,
        last_message_at=row.last_message_at,
        message_count=row.message_count,
    )


def list_sessions(engine: Engine, limit: int = 50) -> list[SessionInfo]:
    """List all chat sessions ordered by last message time.

    Args:
        engine: SQLAlchemy database engine.
        limit: Maximum number of sessions to return.

    Returns:
        List of session information objects.

    """
    sql = text(
        """
        SELECT id, name, model, first_message_preview, created_at, last_message_at, message_count
        FROM chat_sessions
        ORDER BY last_message_at DESC
        LIMIT :limit
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(sql, {"limit": limit}).fetchall()
    return [
        SessionInfo(
            id=r.id,
            name=r.name,
            model=r.model,
            first_message_preview=r.first_message_preview,
            created_at=r.created_at,
            last_message_at=r.last_message_at,
            message_count=r.message_count,
        )
        for r in rows
    ]


def get_session(engine: Engine, session_id: str) -> SessionDetailResponse | None:
    """Retrieve a session by its ID with full message history.

    Args:
        engine: SQLAlchemy database engine.
        session_id: Unique identifier of the session.

    Returns:
        Session details including messages, or None if not found.

    """
    sql = text(
        """
        SELECT id, name, model, first_message_preview, created_at, last_message_at, message_count, messages
        FROM chat_sessions
        WHERE id = :session_id
        """
    )
    with engine.connect() as conn:
        row = conn.execute(sql, {"session_id": session_id}).fetchone()
    if row is None:
        return None
    messages_data = row.messages or []
    messages = [MessageRecord(**m) for m in messages_data]
    return SessionDetailResponse(
        id=row.id,
        name=row.name,
        model=row.model,
        first_message_preview=row.first_message_preview,
        created_at=row.created_at,
        last_message_at=row.last_message_at,
        message_count=row.message_count,
        messages=messages,
    )


def update_session_name(engine: Engine, session_id: str, name: str) -> None:
    """Update the name of an existing session.

    Args:
        engine: SQLAlchemy database engine.
        session_id: Unique identifier of the session.
        name: New name for the session.

    """
    sql = text("UPDATE chat_sessions SET name = :name WHERE id = :session_id")
    with engine.begin() as conn:
        conn.execute(sql, {"name": name, "session_id": session_id})


def delete_session(engine: Engine, session_id: str) -> None:
    """Delete a session from the database.

    Args:
        engine: SQLAlchemy database engine.
        session_id: Unique identifier of the session to delete.

    """
    sql = text("DELETE FROM chat_sessions WHERE id = :session_id")
    with engine.begin() as conn:
        conn.execute(sql, {"session_id": session_id})


def append_message(engine: Engine, session_id: str, role: str, content: str) -> int:
    """Append a message to a session's history.

    Args:
        engine: SQLAlchemy database engine.
        session_id: Unique identifier of the session.
        role: Role of the message sender (user/assistant).
        content: Content of the message.

    """
    now = datetime.now(timezone.utc).isoformat()
    msg_json = json.dumps([{"role": role, "content": content, "timestamp": now}])
    sql = text(
        """
        UPDATE chat_sessions
        SET messages = messages || CAST(:msg AS JSONB),
            message_count = message_count + 1,
            last_message_at = NOW()
        WHERE id = :session_id
        RETURNING message_count
        """
    )
    with engine.begin() as conn:
        row = conn.execute(sql, {"msg": msg_json, "session_id": session_id}).fetchone()
    return row[0] if row else 0


def touch_session(engine: Engine, session_id: str) -> None:
    """Update last_message_at for a session (used after tool calls or activity)."""
    sql = text(
        "UPDATE chat_sessions SET last_message_at = NOW() WHERE id = :session_id"
    )
    with engine.begin() as conn:
        conn.execute(sql, {"session_id": session_id})
