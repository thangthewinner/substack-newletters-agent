import os
import uuid
from collections.abc import Generator

import gradio as gr
import requests
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")
API_BASE_URL = BACKEND_URL
SESSION_ID = str(uuid.uuid4())


def stream_chat(
    messages: list[dict[str, str]],
) -> Generator[str, None, None]:
    """Stream chat responses from the backend chat endpoint."""
    payload = {
        "messages": messages,
        "session_id": SESSION_ID,
    }

    try:
        with requests.post(
            f"{API_BASE_URL}/chat/stream", json=payload, stream=True
        ) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:
                    yield chunk
    except requests.RequestException as exc:
        yield f"Error: {exc}"


def build_messages(
    history: list[dict[str, str] | tuple[str | None, str | None] | list[str | None]],
    message: str,
) -> list[dict[str, str]]:
    """Build backend chat payload from Gradio history + latest user message."""
    messages: list[dict[str, str]] = []
    for item in history:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant"} and isinstance(content, str):
                messages.append({"role": role, "content": content})
            continue
        if isinstance(item, (list, tuple)) and len(item) == 2:
            user_msg, assistant_msg = item
            if isinstance(user_msg, str):
                messages.append({"role": "user", "content": user_msg})
            if isinstance(assistant_msg, str):
                messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})
    return messages


def chat(
    message: str,
    history: list[dict[str, str] | tuple[str | None, str | None] | list[str | None]],
) -> Generator[str, None, None]:
    """Handle chat messages with streaming response."""
    if not message.strip():
        yield "Please enter a message."
        return

    messages = build_messages(history, message)

    response = ""
    for chunk in stream_chat(messages):
        response += chunk
        yield response


with gr.Blocks(title="Substack Chatbot", theme=gr.themes.Soft()) as demo:
    gr.HTML(
        "<div style='background-color:#ff6719; padding:20px; border-radius:12px; "
        "text-align:center; margin-bottom:20px;'>\n"
        "    <h1 style='color:white; font-size:42px; font-family:serif; margin:0;'>\n"
        "        Substack Newsletter Chatbot\n"
        "    </h1>\n"
        "</div>\n"
    )

    chatbot = gr.Chatbot(height=700)

    gr.ChatInterface(
        fn=chat,
        chatbot=chatbot,
        examples=[
            ["What are the latest AI trends in 2025?"],
            ["List articles about LLMs"],
            ["What was published on 2025-03-20?"],
        ],
    )


if __name__ == "__main__":
    demo.launch()
