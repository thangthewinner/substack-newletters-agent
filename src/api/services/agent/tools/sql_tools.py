"""Sql Tools."""
import asyncio
from datetime import datetime, timedelta

from langchain_core.tools import BaseTool, tool
from sqlalchemy import func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from src.models.sql_models import SubstackArticle


def create_sql_tools(db_engine: Engine) -> list[BaseTool]:
    """Create SQL-backed tools used by the chat agent."""

    def _validate_period_inputs(
        year: int | None,
        month: int | None,
        start_date: str | None,
        end_date: str | None,
        limit: int | None = None,
    ) -> None:
        if month is not None and (month < 1 or month > 12):
            raise ValueError("month must be between 1 and 12.")
        if year is not None and year < 1:
            raise ValueError("year must be a positive integer.")
        if limit is not None and limit < 1:
            raise ValueError("limit must be greater than 0.")
        if start_date:
            datetime.strptime(start_date, "%Y-%m-%d")
        if end_date:
            datetime.strptime(end_date, "%Y-%m-%d")
        if start_date and end_date and start_date > end_date:
            raise ValueError("start_date must be less than or equal to end_date.")

    def _apply_period_filters(
        stmt,
        year: int | None,
        month: int | None,
        start_date: str | None,
        end_date: str | None,
        feed_name: str | None,
    ):
        if feed_name:
            stmt = stmt.where(SubstackArticle.feed_name == feed_name)
        if year is not None:
            stmt = stmt.where(
                func.extract("year", SubstackArticle.published_at) == year
            )
        if month is not None:
            stmt = stmt.where(
                func.extract("month", SubstackArticle.published_at) == month
            )
        if start_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            stmt = stmt.where(SubstackArticle.published_at >= start_dt)
        if end_date:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
            stmt = stmt.where(SubstackArticle.published_at < end_dt)
        return stmt

    def _list_articles_by_period(
        year: int | None,
        month: int | None,
        start_date: str | None,
        end_date: str | None,
        feed_name: str | None,
        limit: int,
    ) -> list[dict[str, str]]:
        _validate_period_inputs(year, month, start_date, end_date, limit)
        with Session(db_engine) as session:
            stmt = select(SubstackArticle)
            stmt = _apply_period_filters(
                stmt=stmt,
                year=year,
                month=month,
                start_date=start_date,
                end_date=end_date,
                feed_name=feed_name,
            )
            stmt = stmt.order_by(SubstackArticle.published_at.desc()).limit(limit)
            rows = session.execute(stmt).scalars().all()
        return [
            {
                "title": row.title,
                "feed_name": row.feed_name,
                "feed_author": row.feed_author,
                "url": row.url,
                "published_at": str(row.published_at),
            }
            for row in rows
        ]

    def _count_articles_by_period(
        year: int | None,
        month: int | None,
        start_date: str | None,
        end_date: str | None,
        feed_name: str | None,
    ) -> int:
        _validate_period_inputs(year, month, start_date, end_date)
        with Session(db_engine) as session:
            stmt = select(func.count(SubstackArticle.id))
            stmt = _apply_period_filters(
                stmt=stmt,
                year=year,
                month=month,
                start_date=start_date,
                end_date=end_date,
                feed_name=feed_name,
            )
            return int(session.execute(stmt).scalar_one())

    def _count_articles_grouped_by_period(
        group_by: str,
        year: int | None,
        feed_name: str | None,
    ) -> list[dict[str, int | str]]:
        if group_by not in {"month", "year"}:
            raise ValueError("group_by must be either 'month' or 'year'.")
        if year is not None and year < 1:
            raise ValueError("year must be a positive integer.")

        if group_by == "month":
            period_expr = func.to_char(SubstackArticle.published_at, "YYYY-MM")
        else:
            period_expr = func.to_char(SubstackArticle.published_at, "YYYY")

        with Session(db_engine) as session:
            stmt = (
                select(
                    period_expr.label("period"),
                    func.count(SubstackArticle.id).label("count"),
                )
                .select_from(SubstackArticle)
                .group_by(period_expr)
                .order_by(period_expr)
            )
            if year is not None:
                stmt = stmt.where(
                    func.extract("year", SubstackArticle.published_at) == year
                )
            if feed_name:
                stmt = stmt.where(SubstackArticle.feed_name == feed_name)

            rows = session.execute(stmt).all()

        return [{"period": str(row.period), "count": int(row.count)} for row in rows]

    @tool
    async def list_articles_by_period(
        year: int | None = None,
        month: int | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        feed_name: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, str]]:
        """List articles by year, month, month+year, or date range.

        Use for requests like 'list articles in 2025', 'articles in July 2025',
        or 'articles from 2025-07-01 to 2025-07-31'.
        """
        return await asyncio.to_thread(
            _list_articles_by_period,
            year,
            month,
            start_date,
            end_date,
            feed_name,
            limit,
        )

    @tool
    async def count_articles_by_period(
        year: int | None = None,
        month: int | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        feed_name: str | None = None,
    ) -> int:
        """Count articles by year, month, month+year, or date range.

        Use for requests like 'how many articles in August', 'in 2025',
        or 'between two dates'.
        """
        return await asyncio.to_thread(
            _count_articles_by_period,
            year,
            month,
            start_date,
            end_date,
            feed_name,
        )

    @tool
    async def count_articles_grouped_by_period(
        group_by: str,
        year: int | None = None,
        feed_name: str | None = None,
    ) -> list[dict[str, int | str]]:
        """Count articles grouped by month or year.

        Use when the user asks for trends such as 'count by month in 2025'.
        """
        return await asyncio.to_thread(
            _count_articles_grouped_by_period,
            group_by,
            year,
            feed_name,
        )

    return [
        list_articles_by_period,
        count_articles_by_period,
        count_articles_grouped_by_period,
    ]
