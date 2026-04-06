"""Create Db."""
from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError

from src.infrastructure.supabase.init_session import init_engine
from src.models.sql_models import Base, ChatSession, SubstackArticle
from src.utils.logger_util import setup_logging

logger = setup_logging()


def create_table() -> None:
    """Create all tables (SubstackArticle, ChatSession) in the Supabase Postgres database.

    This function initializes a SQLAlchemy engine, checks which tables exist,
    and creates any missing ones via Base.metadata.create_all().
    The engine is properly disposed of after the operation to prevent resource leaks.
    Errors during table creation are logged and handled gracefully.
    """
    engine = init_engine()
    try:
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()

        for table_name in [SubstackArticle.__tablename__, ChatSession.__tablename__]:
            if table_name in existing_tables:
                logger.info(f"Table '{table_name}' already exists. No action needed.")
            else:
                logger.info(f"Table '{table_name}' does not exist. Creating...")

        Base.metadata.create_all(bind=engine)
        logger.info("All tables ensured (created or already exist).")

        # Ensure chat_sessions.id has gen_random_uuid() server default
        # (needed for tables created before this column default was added)
        if "chat_sessions" in existing_tables:
            try:
                with engine.begin() as conn:
                    conn.execute(
                        text(
                            "ALTER TABLE chat_sessions ALTER COLUMN id SET DEFAULT gen_random_uuid();"
                        )
                    )
                logger.info("Ensured chat_sessions.id has gen_random_uuid() default.")
            except SQLAlchemyError:
                pass  # Already set or table doesn't exist yet
    except SQLAlchemyError as e:
        logger.exception("SQLAlchemy error creating tables")
        raise SQLAlchemyError("Failed to create tables") from e
    except Exception:
        logger.exception("Unexpected error creating tables")
        raise
    finally:
        engine.dispose()
        logger.info("Database engine disposed.")


if __name__ == "__main__":
    create_table()
