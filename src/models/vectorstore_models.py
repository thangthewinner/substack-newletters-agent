"""Vectorstore Models."""
from datetime import UTC, datetime

from pydantic import BaseModel, Field, HttpUrl


class ArticleChunkPayload(BaseModel):
    """Pydantic model for article chunk metadata stored in Qdrant.

    This model defines the payload structure for each vector point in Qdrant,
    containing both the original article metadata and the chunk-specific data.

    Attributes:
        feed_name: Name of the Substack feed.
        feed_author: Author of the Substack feed.
        article_authors: List of article authors.
        title: Article title.
        url: Article URL.
        published_at: Publication date.
        created_at: Record creation date.
        chunk_index: Index of this chunk within the article.
        chunk_text: Text content of this chunk.

    """

    feed_name: str = Field(default="", description="Name of the feed")
    feed_author: str = Field(default="", description="Author of the feed")
    article_authors: list[str] = Field(
        default_factory=list, description="Authors of the article"
    )
    title: str = Field(default="", description="Title of the article")
    url: HttpUrl | str | None = Field(default=None, description="URL of the article")
    published_at: datetime | str = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Publication date of the article",
    )
    created_at: datetime | str = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Creation date of the article",
    )
    chunk_index: int = Field(default=0, description="Index of the article chunk")
    chunk_text: str | None = Field(
        default=None, description="Text content of the article chunk"
    )
