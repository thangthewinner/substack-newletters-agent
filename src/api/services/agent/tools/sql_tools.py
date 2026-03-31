from langchain_core.tools import BaseTool, tool
from sqlalchemy import func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from src.models.sql_models import SubstackArticle


def create_sql_tools(db_engine: Engine) -> list[BaseTool]:
    """Create SQL-backed tools used by the chat agent."""

    @tool
    async def get_articles_by_date(
        date: str,
        feed_name: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, str]]:
        """Fetch articles published on a specific date (format: YYYY-MM-DD).
        Use when the user asks 'articles from [date]' or 'what was published on [date]'.
        Example: date='2025-03-20'"""
        with Session(db_engine) as session:
            stmt = select(SubstackArticle).where(
                func.date(SubstackArticle.published_at) == date
            )
            if feed_name:
                stmt = stmt.where(SubstackArticle.feed_name == feed_name)
            stmt = stmt.order_by(SubstackArticle.published_at.desc()).limit(limit)
            rows = session.execute(stmt).scalars().all()

        return [
            {
                "title": r.title,
                "feed_name": r.feed_name,
                "feed_author": r.feed_author,
                "url": r.url,
                "published_at": str(r.published_at),
            }
            for r in rows
        ]

    return [get_articles_by_date]
