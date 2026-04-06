"""Naming Service."""
import asyncio

from langchain_core.messages import HumanMessage, SystemMessage

from src.api.services.agent.llm_factory import create_agent_llm
from src.config import settings
from src.utils.logger_util import setup_logging

logger = setup_logging()

NAMING_PROMPT = "Summarize this question in 5 words or less for a chat title:"


async def generate_session_name(first_message: str) -> str:
    """Generate a short session name from the first user message.

    Args:
        first_message: The first user message in the session.

    Returns:
        A short session name (max ~5 words), or "New Chat" on failure.

    """
    try:
        llm = create_agent_llm(settings.agent.naming_model)
        messages = [
            SystemMessage(content=NAMING_PROMPT),
            HumanMessage(content=first_message),
        ]
        response = await asyncio.wait_for(
            llm.ainvoke(messages),
            timeout=15.0,
        )
        content = response.content
        if isinstance(content, list):
            # Multimodal models may return content as a list of parts
            content = " ".join(c for c in content if isinstance(c, str))
        name = str(content).strip().strip('"').strip("'").strip()
        return name if name else "New Chat"
    except asyncio.TimeoutError:
        logger.warning("Session naming timed out, using default name")
    except Exception:
        logger.exception("Failed to generate session name")
    return "New Chat"
