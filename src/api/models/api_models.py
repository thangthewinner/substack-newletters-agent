from pydantic import BaseModel, Field


# Core search result model
class SearchResult(BaseModel):
    title: str = Field(default="", description="Title of the article")
    feed_author: str | None = Field(default=None, description="Author of the article")
    feed_name: str | None = Field(
        default=None, description="Name of the feed/newsletter"
    )
    article_author: list[str] | None = Field(
        default=None, description="List of article authors"
    )
    url: str | None = Field(
        default=None, description="Text content of the article chunk"
    )
    chunk_text: str | None = Field(
        default=None, description="Text content of the article chunk"
    )
    score: float = Field(default=0.0, description="Relevance score of the article")


# Unique titles request/response
class UniqueTitleRequest(BaseModel):
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
    results: list[SearchResult] = Field(
        default_factory=list, description="List of unique title search results"
    )


# Chat request/response models
class ChatMessage(BaseModel):
    role: str = Field(description="Role: 'user' or 'assistant'")
    content: str = Field(description="Message content")


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(description="Full conversation history")


class ChatResponse(BaseModel):
    reply: str = Field(description="Agent's final answer in Markdown")
