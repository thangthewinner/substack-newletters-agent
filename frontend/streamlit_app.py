"""Streamlit frontend for Substack newsletter search chatbot."""

import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")
API = f"{BACKEND_URL}"

st.set_page_config(page_title="Substack Chatbot", layout="wide")


def api_get(path: str, params: dict | None = None):
    """Send GET request to backend API.

    Args:
        path: API endpoint path.
        params: Optional query parameters.

    Returns:
        JSON response from the API.

    """
    r = requests.get(f"{API}{path}", params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def api_post(path: str, json: dict | None = None):
    """Send POST request to backend API.

    Args:
        path: API endpoint path.
        json: Optional JSON body payload.

    Returns:
        JSON response from the API.

    """
    r = requests.post(f"{API}{path}", json=json, timeout=30)
    r.raise_for_status()
    return r.json()


def api_delete(path: str):
    """Send DELETE request to backend API.

    Args:
        path: API endpoint path.

    Returns:
        JSON response from the API.

    """
    r = requests.delete(f"{API}{path}", timeout=10)
    r.raise_for_status()
    return r.json()


def load_sessions():
    """Load all chat sessions from the backend.

    Returns:
        List of session objects, or empty list on error.

    """
    try:
        return api_get("/sessions")
    except Exception:
        return []


def load_session_detail(session_id: str):
    """Load detailed information for a specific session.

    Args:
        session_id: The unique identifier of the session.

    Returns:
        Session details including messages, or None on error.

    """
    try:
        return api_get(f"/sessions/{session_id}")
    except Exception:
        return None


def create_session(name: str = "New Chat"):
    """Create a new chat session.

    Args:
        name: Optional name for the new session.

    Returns:
        Created session object with id.

    """
    return api_post("/sessions", json={"name": name})


def delete_session(session_id: str):
    """Delete a chat session.

    Args:
        session_id: The unique identifier of the session to delete.

    """
    api_delete(f"/sessions/{session_id}")


def stream_chat(session_id: str, messages: list[dict]):
    """Stream chat messages to backend and yield responses.

    Args:
        session_id: The unique identifier of the session.
        messages: List of message objects with role and content.

    Yields:
        Response chunks from the backend.

    """
    payload = {"messages": messages, "session_id": session_id}
    with requests.post(
        f"{API}/chat/stream", json=payload, stream=True, timeout=180
    ) as r:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                yield chunk


def init_state():
    """Initialize Streamlit session state with default values."""
    if "sessions" not in st.session_state:
        st.session_state.sessions = load_sessions()
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []


def switch_session(session_id: str):
    """Switch to a different chat session.

    Args:
        session_id: The unique identifier of the session to switch to.

    """
    st.session_state.current_session_id = session_id
    detail = load_session_detail(session_id)
    if detail and detail.get("messages"):
        st.session_state.messages = [
            {"role": m["role"], "content": m["content"]} for m in detail["messages"]
        ]
    else:
        st.session_state.messages = []


def refresh_sessions():
    """Reload sessions from backend into session state."""
    st.session_state.sessions = load_sessions()


def render_sidebar():
    """Render the sidebar with session list and controls."""
    with st.sidebar:
        st.markdown("### 🟠 Substack Chatbot")

        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            try:
                new_sess = create_session()
                refresh_sessions()
                switch_session(new_sess["id"])
                st.rerun()
            except requests.RequestException as e:
                st.error(f"Failed to create session: {e}")

        st.divider()
        st.markdown("### 📋 Sessions")

        for sess in st.session_state.sessions:
            sid = sess["id"]
            name = sess.get("name", "New Chat")
            is_active = sid == st.session_state.current_session_id

            col_name, col_del = st.columns([5, 1])

            with col_name:
                label = f"💬 {name}" if is_active else f"• {name}"
                if st.button(label, key=f"sess_{sid}", use_container_width=True):
                    if not is_active:
                        switch_session(sid)
                        st.rerun()

            with col_del:
                if st.button("🗑️", key=f"del_{sid}", help="Delete session"):
                    try:
                        delete_session(sid)
                        refresh_sessions()
                        if sid == st.session_state.current_session_id:
                            st.session_state.current_session_id = None
                            st.session_state.messages = []
                        st.rerun()
                    except requests.RequestException as e:
                        st.error(f"Failed to delete: {e}")


def render_main():
    """Render the main chat interface with messages and input."""
    if not st.session_state.current_session_id:
        if st.session_state.sessions:
            switch_session(st.session_state.sessions[0]["id"])
            st.rerun()
        else:
            st.info("No sessions yet. Click **New Chat** in the sidebar to start.")
            return

    current = next(
        (
            s
            for s in st.session_state.sessions
            if s["id"] == st.session_state.current_session_id
        ),
        None,
    )
    title = current["name"] if current else "Chat"
    st.markdown(f"### 🗨️ {title}")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)

    if prompt := st.chat_input("Type your message..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        messages_payload = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]

        with st.chat_message("assistant"):
            full_reply = ""
            response_placeholder = st.empty()
            try:
                for chunk in stream_chat(
                    st.session_state.current_session_id, messages_payload
                ):
                    if chunk.startswith("__TOOL_START__:"):
                        tname = chunk.split(":", 1)[1].strip()
                        st.toast(f"🛠️ Using tool: `{tname}`...")
                        continue
                    full_reply += chunk
                    response_placeholder.markdown(full_reply, unsafe_allow_html=True)
            except requests.RequestException as e:
                error_msg = f"Error: {e}"
                response_placeholder.markdown(error_msg, unsafe_allow_html=True)
                full_reply = error_msg

        st.session_state.messages.append({"role": "assistant", "content": full_reply})
        refresh_sessions()
        st.rerun()


def main():
    """Initialize state, render sidebar, and run the main interface."""
    init_state()
    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()
