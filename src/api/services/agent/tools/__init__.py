from langchain_core.tools import BaseTool
from sqlalchemy.engine import Engine

from src.api.services.agent.tools.search_tools import create_search_tools
from src.api.services.agent.tools.sql_tools import create_sql_tools
from src.infrastructure.qdrant.qdrant_vectorstore import AsyncQdrantVectorStore


def create_tools(
    vectorstore: AsyncQdrantVectorStore, db_engine: Engine
) -> list[BaseTool]:
    """
    Create all agent tools, injecting required infrastructure via closure.

    Args:
        vectorstore: Qdrant vector store instance from app.state.
        db_engine: SQLAlchemy engine for Supabase queries.

    Returns:
        list: All registered LangChain tools.
    """
    search_tools = create_search_tools(vectorstore)
    sql_tools = create_sql_tools(db_engine)
    return [*search_tools, *sql_tools]
