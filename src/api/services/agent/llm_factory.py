from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from src.config import settings


def create_agent_llm(model: str | None = None):
    """
    Initialize a ChatOpenAI instance routed through OpenRouter.

    Args:
        model: Optional model override. Defaults to settings.agent.default_model.

    Returns:
        ChatOpenAI: Configured LLM instance with tool-calling support.
    """
    agent_settings = settings.agent
    raw_api_key = settings.openrouter.api_key
    raw_value = (
        raw_api_key.get_secret_value()
        if isinstance(raw_api_key, SecretStr)
        else str(raw_api_key)
    )
    api_key = SecretStr(raw_value)
    llm = ChatOpenAI(
        base_url=settings.openrouter.api_url,
        api_key=api_key,
        model=model or agent_settings.default_model,
        temperature=agent_settings.temperature,
        max_completion_tokens=agent_settings.max_tokens,
        max_retries=agent_settings.max_retries,
    )
    return llm
