"""Llm Factory."""
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from src.config import settings


def create_agent_llm(model: str | None = None):
    """Initialize an LLM instance based on provider prefix in model string.

    Format: {provider}/{model_name}
    Providers: openrouter, groq, openai, anthropic

    Args:
        model: Optional model override. Defaults to settings.agent.default_model.

    Returns:
        BaseChatModel: Configured LLM instance with tool-calling support.

    """
    model_str = model or settings.agent.default_model
    agent_settings = settings.agent

    if "/" in model_str:
        provider, model_name = model_str.split("/", 1)
    else:
        provider = "openrouter"
        model_name = model_str

    supported = {"openrouter", "groq", "openai", "anthropic"}
    if provider not in supported:
        raise ValueError(
            f"Unsupported LLM provider: '{provider}'. "
            f"Supported providers: {sorted(supported)}. "
            f"Model string format: '{{provider}}/{{model_name}}'"
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(  # type: ignore[call-arg]
            model=model_name,
            api_key=SecretStr(settings.anthropic.api_key),
            temperature=agent_settings.temperature,
            max_tokens=agent_settings.max_tokens,
        )

    base_urls = {
        "openrouter": settings.openrouter.api_url,
        "groq": "https://api.groq.com/openai/v1",
        "openai": None,
    }
    api_keys = {
        "openrouter": settings.openrouter.api_key,
        "groq": settings.groq.api_key,
        "openai": settings.openai.api_key,
    }

    raw_key = api_keys.get(provider, "")
    api_key = SecretStr(
        raw_key
        if isinstance(raw_key, str)
        else raw_key.get_secret_value()
        if isinstance(raw_key, SecretStr)
        else str(raw_key)
    )

    return ChatOpenAI(
        model=model_name,
        base_url=base_urls.get(provider),
        api_key=api_key,
        temperature=agent_settings.temperature,
        max_completion_tokens=agent_settings.max_tokens,
        max_retries=agent_settings.max_retries,
    )
