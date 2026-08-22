from dotenv import dotenv_values, find_dotenv

import streamlit as st

GEMINI_KEY_NAME = "gemini_api_key"
HUGGINGFACE_KEY_NAME = "huggingface_api_key"


def load_dotenv_values() -> dict:
    """Read a local .env if present; never raises if it's missing."""
    path = find_dotenv(usecwd=True)
    if not path:
        return {}
    return dict(dotenv_values(path))


def resolve_initial_key(existing: str | None, dotenv_value: str | None) -> str:
    if existing:
        return existing
    if dotenv_value:
        return dotenv_value
    return ""


def init_credential_state(session_state: dict, dotenv_values: dict | None = None) -> None:
    """Seed session_state key fields exactly once; safe to call every rerun."""
    if dotenv_values is None:
        dotenv_values = load_dotenv_values()

    if GEMINI_KEY_NAME not in session_state:
        session_state[GEMINI_KEY_NAME] = resolve_initial_key(
            None, dotenv_values.get("GEMINI_API_KEY")
        )
    if HUGGINGFACE_KEY_NAME not in session_state:
        session_state[HUGGINGFACE_KEY_NAME] = resolve_initial_key(
            None, dotenv_values.get("HUGGINGFACE_API_KEY")
        )


def get_gemini_key(session_state: dict) -> str:
    return session_state.get(GEMINI_KEY_NAME, "")


def get_huggingface_key(session_state: dict) -> str:
    return session_state.get(HUGGINGFACE_KEY_NAME, "")


def missing_key_message(task_label: str, provider_label: str) -> str:
    return f"Enter your {provider_label} API key in the sidebar to use {task_label}."


def render_key_sidebar() -> None:
    """Streamlit wiring — manual/integration verified only (Task 4)."""
    init_credential_state(st.session_state)

    st.sidebar.text_input(
        "Gemini API key", type="password", key=GEMINI_KEY_NAME
    )
    st.sidebar.text_input(
        "HuggingFace API key", type="password", key=HUGGINGFACE_KEY_NAME
    )

    gemini_status = "✅ Gemini connected" if get_gemini_key(st.session_state) else "⚠️ Gemini key missing"
    hf_status = "✅ HuggingFace connected" if get_huggingface_key(st.session_state) else "⚠️ HuggingFace key missing"
    st.sidebar.caption(gemini_status)
    st.sidebar.caption(hf_status)
