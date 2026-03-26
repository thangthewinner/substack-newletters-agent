import uuid
from uuid import UUID

from sqlalchemy import ARRAY, TIMESTAMP, BigInteger, String, Text, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from src.config import settings


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy declarative models."""

    pass


class SubstackArticle(Base):
    """SQLAlchemy model representing a Substack newsletter article.

    Stores article metadata and content fetched from RSS feeds.
    Each article is uniquely identified by its URL and has a generated UUID.

    Attributes:
        id: Primary key - auto-incrementing integer.
        uuid: Unique UUID for external references.
        feed_name: Name of the Substack newsletter/feed.
        feed_author: Author of the newsletter/feed.
        article_authors: List of article-specific authors.
        title: Article title.
        url: Canonical URL of the article.
        content: Full article content in Markdown format.
        published_at: Publication timestamp.
        created_at: Record creation timestamp (server-set).
    """

    __tablename__ = settings.supabase_db.table_name

    # Primary internal ID
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, index=True)

    # External unique identifier
    uuid: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        default=uuid.uuid4,
        unique=True,
        nullable=False,
        index=True,
    )

    # Article fields
    feed_name: Mapped[str] = mapped_column(String, nullable=False)
    feed_author: Mapped[str] = mapped_column(String, nullable=False)
    article_authors: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False)
    title: Mapped[str] = mapped_column(String, nullable=False)
    url: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    published_at: Mapped[str] = mapped_column(TIMESTAMP, nullable=False)
    created_at: Mapped[str] = mapped_column(
        TIMESTAMP, server_default=func.now(), nullable=False
    )
