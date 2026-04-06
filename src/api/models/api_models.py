"""Pydantic models for API request/response schemas."""

from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class SearchResult(BaseModel):
    """Represents a single search result from Qdrant vector store."""

    title: str = Field(default="", description="Title of the article")
    feed_author: str | None = Field(default=None, description="Author of the article")
    feed_name: str | None = Field(
        default=None, description="Name of the feed/newsletter"
    )
    article_author: list[str] | None = Field(
        default=None, description="List of article authors"
    )
    url: str | None = Field(default=None, description="URL of the article")
    chunk_text: str | None = Field(
        default=None, description="Text content of the article chunk"
    )
    score: float = Field(default=0.0, description="Relevance score of the article")


class UniqueTitleRequest(BaseModel):
    """Request body for the unique titles search endpoint."""

    query_text: str = Field(default="", description="The user query text")
    feed_author: str | None = Field(default=None, description="Filter by author name")
    feed_name: str | None = Field(
        default=None, description="Filter by feed/newsletter name"
    )
    article_author: list[str] | None = Field(
        default=None, description="List of article authors"
    )
    title_keywords: str | None = Field(
        default=None, description="Keywords or phrase to match in title"
    )
    limit: int = Field(default=5, description="Number of results to return")


class UniqueTitleResponse(BaseModel):
    """Response body containing unique article titles."""

    results: list[SearchResult] = Field(
        default_factory=list, description="List of unique title search results"
    )


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""

    role: str = Field(description="Role: 'user' or 'assistant'")
    content: str = Field(description="Message content")


class ChatRequest(BaseModel):
    """Request body for the chat endpoint."""

    messages: list[ChatMessage] = Field(description="Full conversation history")
    session_id: str = Field(
        description="Conversation session ID for server-side memory",
    )

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        """Validate that session_id is a valid UUID format."""
        UUID(v)  # Raises ValueError if invalid
        return v


class ChatResponse(BaseModel):
    """Response body containing the agent's reply."""

    reply: str = Field(description="Agent's final answer in Markdown")
