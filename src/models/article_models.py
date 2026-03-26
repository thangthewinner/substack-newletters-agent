from pydantic import BaseModel, Field, field_validator


# Feed Settings
class FeedItem(BaseModel):
    name: str = Field(default="", description="Name of the feed")
    author: str = Field(default="", description="Author of the feed")
    url: str = Field(default="", description="URL of the feed")

    @field_validator("name", "author", "url", mode="before")
    @classmethod
    def _strip_unicode_quotes_and_whitespace(cls, v: object) -> object:
        if not isinstance(v, str):
            return v
        s = v.strip()
        # Handle common “smart quotes” and normal quotes from copy/paste.
        # Repeat until stable to handle nested quoting.
        quote_chars = "\"'“”‘’"
        while len(s) >= 2 and (s[0] in quote_chars) and (s[-1] in quote_chars):
            s = s[1:-1].strip()
        return s


# Article settings
class ArticleItem(BaseModel):
    feed_name: str = Field(default="", description="Name of the feed")
    feed_author: str = Field(default="", description="Author of the feed")
    title: str = Field(default="", description="Title of the article")
    url: str = Field(default="", description="URL of the article")
    content: str = Field(default="", description="Content of the article")
    article_authors: list[str] = Field(
        default_factory=list, description="Authors of the article"
    )
    published_at: str | None = Field(
        default=None, description="Publication date of the article"
    )
