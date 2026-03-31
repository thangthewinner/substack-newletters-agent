from langchain_core.tools import BaseTool, tool

from src.api.services import search_service
from src.infrastructure.qdrant.qdrant_vectorstore import AsyncQdrantVectorStore


def create_search_tools(vectorstore: AsyncQdrantVectorStore) -> list[BaseTool]:
    """Create Qdrant-backed tools used by the chat agent."""

    @tool
    async def search_articles(
        query: str,
        feed_name: str | None = None,
        feed_author: str | None = None,
        title_keywords: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, object]]:
        """Search newsletter articles by semantic similarity.
        Use for general topic or concept queries (e.g. 'AI trends in 2025')."""
        results = await search_service.query_with_filters(
            vectorstore=vectorstore,
            query_text=query,
            feed_name=feed_name,
            feed_author=feed_author,
            title_keywords=title_keywords,
            limit=limit,
        )
        return [r.model_dump() for r in results]

    @tool
    async def search_unique_titles(
        query: str,
        feed_name: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, object]]:
        """Return unique article titles matching a topic.
        Use when the user wants to discover or list available articles."""
        results = await search_service.query_unique_titles(
            vectorstore=vectorstore,
            query_text=query,
            feed_name=feed_name,
            limit=limit,
        )
        return [r.model_dump() for r in results]

    return [search_articles, search_unique_titles]
