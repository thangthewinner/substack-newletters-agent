"""Search Service."""
from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchText,
    MatchValue,
    Prefetch,
)

from src.api.models.api_models import SearchResult
from src.infrastructure.qdrant.qdrant_vectorstore import AsyncQdrantVectorStore
from src.utils.logger_util import setup_logging

logger = setup_logging()


async def _execute_qdrant_query(
    vectorstore: AsyncQdrantVectorStore,
    query_text: str,
    feed_author: str | None,
    feed_name: str | None,
    title_keywords: str | None,
    fetch_limit: int,
):
    dense_vector = vectorstore.dense_vectors([query_text])[0]
    sparse_vector = vectorstore.sparse_vectors([query_text])[0]

    # Build filter conditions
    conditions: list[FieldCondition] = []
    if feed_author:
        conditions.append(
            FieldCondition(key="feed_author", match=MatchValue(value=feed_author))
        )
    if feed_name:
        conditions.append(
            FieldCondition(key="feed_name", match=MatchValue(value=feed_name))
        )
    if title_keywords:
        conditions.append(
            FieldCondition(
                key="title", match=MatchText(text=title_keywords.strip().lower())
            )
        )

    query_filter = Filter(must=conditions) if conditions else None  # type: ignore
    logger.info(f"Fetching up to {fetch_limit} points.")

    return await vectorstore.client.query_points(
        collection_name=vectorstore.collection_name,
        query=FusionQuery(fusion=Fusion.RRF),
        prefetch=[
            Prefetch(
                query=dense_vector,
                using="Dense",
                limit=fetch_limit,
                filter=query_filter,
            ),
            Prefetch(
                query=sparse_vector,
                using="Sparse",
                limit=fetch_limit,
                filter=query_filter,
            ),
        ],
        query_filter=query_filter,
        limit=fetch_limit,
    )


async def query_with_filters(
    vectorstore: AsyncQdrantVectorStore,
    query_text: str = "",
    feed_author: str | None = None,
    feed_name: str | None = None,
    title_keywords: str | None = None,
    limit: int = 5,
) -> list[SearchResult]:
    """Query the vector store with optional filters and return search results.

    Performs a hybrid dense + sparse search on Qdrant and applies filters based
    on feed author, feed name, and title keywords. Results are deduplicated by point ID.
    """
    # Fetch 100x the requested limit to account for deduplication by point ID
    fetch_limit = max(1, limit) * 100
    
    response = await _execute_qdrant_query(
        vectorstore=vectorstore,
        query_text=query_text,
        feed_author=feed_author,
        feed_name=feed_name,
        title_keywords=title_keywords,
        fetch_limit=fetch_limit,
    )

    # Deduplicate by point ID
    seen_ids: set[str] = set()
    results: list[SearchResult] = []
    for point in response.points:
        if point.id in seen_ids:
            continue
        seen_ids.add(point.id)  # type: ignore
        payload = point.payload or {}
        results.append(
            SearchResult(
                title=payload.get("title", ""),
                feed_author=payload.get("feed_author"),
                feed_name=payload.get("feed_name"),
                article_author=payload.get("article_authors"),
                url=payload.get("url"),
                chunk_text=payload.get("chunk_text"),
                score=point.score,
            )
        )

    results = results[:limit]
    logger.info(f"Returning {len(results)} results for matching query '{query_text}'")
    return results


async def query_unique_titles(
    vectorstore: AsyncQdrantVectorStore,
    query_text: str,
    feed_author: str | None = None,
    feed_name: str | None = None,
    title_keywords: str | None = None,
    limit: int = 5,
) -> list[SearchResult]:
    """Query the vector store and return only unique titles.

    Performs a hybrid dense + sparse search with optional filters. Deduplicates results by article title.
    """
    # Fetch 280x the requested limit: articles have many chunks,
    # so we need far more raw points to yield `limit` unique titles
    fetch_limit = max(1, limit) * 280
    
    response = await _execute_qdrant_query(
        vectorstore=vectorstore,
        query_text=query_text,
        feed_author=feed_author,
        feed_name=feed_name,
        title_keywords=title_keywords,
        fetch_limit=fetch_limit,
    )

    # Deduplicate by title
    seen_titles: set[str] = set()
    results: list[SearchResult] = []
    for point in response.points:
        payload = point.payload or {}
        title = payload.get("title")
        if not title or title in seen_titles:
            continue
        seen_titles.add(title)
        results.append(
            SearchResult(
                title=title,
                feed_author=payload.get("feed_author"),
                feed_name=payload.get("feed_name"),
                article_author=payload.get("article_authors"),
                url=payload.get("url"),
                chunk_text=payload.get("chunk_text"),
                score=point.score,
            )
        )
        if len(results) >= limit:
            break

    logger.info(
        f"Returning {len(results)} unique title results for matching query '{query_text}'"
    )

    return results
