"""Pydantic models for chat session management."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class SessionCreate(BaseModel):
    """Request body for creating a new chat session."""

    name: str = "New Chat"
    model: str | None = None


class SessionUpdate(BaseModel):
    """Request body for updating a session's name."""

    name: str


class MessageRecord(BaseModel):
    """A single message within a chat session."""

    role: str
    content: str
    timestamp: str


class SessionInfo(BaseModel):
    """Summary metadata for a chat session."""

    id: UUID
    name: str
    model: str | None
    first_message_preview: str | None
    created_at: datetime
    last_message_at: datetime
    message_count: int


class SessionDetailResponse(SessionInfo):
    """Full session details including all messages."""

    messages: list[MessageRecord] = Field(default_factory=list)
