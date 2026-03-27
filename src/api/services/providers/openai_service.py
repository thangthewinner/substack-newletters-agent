import os
from collections.abc import AsyncGenerator

from openai import AsyncOpenAI
from opik.integrations.openai import track_openai

from src.api.models.provider_models import ModelConfig
from src.api.services.providers.utils.messages import build_messages
from src.config import settings
from src.utils.logger_util import setup_logging

logger = setup_logging()


# OpenAI client
openai_key = settings.openai.api_key
async_openai_client = AsyncOpenAI(api_key=openai_key)

# Opik Observability
os.environ["OPIK_API_KEY"] = settings.opik.api_key
os.environ["OPIK_PROJECT_NAME"] = settings.opik.project_name

async_openai_client = track_openai(async_openai_client)


async def generate_openai(prompt: str, config: ModelConfig) -> tuple[str, None]:
    """
    Generate a response from OpenAI for a given prompt and model configuration.

    Args:
        prompt (str): The input prompt.
        config (ModelConfig): The model configuration.

    Returns:
        tuple[str, None]: The generated response and None for model and finish reason.

    """
    ### NOTES ON PARAMETERS
    # logprobs: Include the log probabilities on the logprobs most likely tokens,
    #   as well the chosen tokens.
    # temperature: 0.0 (more deterministic) to 1.0 (more creative)
    # top_p: 0.0 to 1.0, nucleus sampling, 1.0 means no nucleus sampling
    #   0.1 means only the tokens comprising the top 10% probability mass are considered.
    # presence_penalty: -2.0 to 2.0, positive values penalize new tokens based
    #   on whether they appear in the text so far
    #   (Encourages model to use more context from other chunks)
    # frequency_penalty: -2.0 to 2.0, positive values penalize new tokens based
    #   on their existing frequency in the text so far (helpful if context chunks overlap.)

    resp = await async_openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=build_messages(prompt),
        temperature=config.temperature,
        max_completion_tokens=config.max_completion_tokens,
        # logprobs=True,
        # top_logprobs=3,
        # top_p=1.0,
        # presence_penalty=0.3,
        # frequency_penalty=0.3,
    )

    return resp.choices[0].message.content or "", None


def stream_openai(prompt: str, config: ModelConfig) -> AsyncGenerator[str, None]:
    """
    Stream a response from OpenAI for a given prompt and model configuration.

    Args:
        prompt (str): The input prompt.
        config (ModelConfig): The model configuration.

    Returns:
        AsyncGenerator[str, None]: An asynchronous generator yielding response chunks.

    """

    async def gen() -> AsyncGenerator[str, None]:
        stream = await async_openai_client.chat.completions.create(
            model=config.primary_model,
            messages=build_messages(prompt),
            temperature=config.temperature,
            max_completion_tokens=config.max_completion_tokens,
            stream=True,
        )

        last_finish_reason = None
        async for chunk in stream:
            delta_text = getattr(chunk.choices[0].delta, "content", None)
            if delta_text:
                yield delta_text

            # Reasons: tool_calls, stop, length, content_filter, error
            finish_reason = getattr(chunk.choices[0], "finish_reason", None)

            if finish_reason:
                last_finish_reason = finish_reason

        logger.warning(f"Final finish_reason: {last_finish_reason}")

        # Yield a chunk to trigger truncation warning in UI
        if last_finish_reason == "length":
            yield "__truncated__"

    return gen()