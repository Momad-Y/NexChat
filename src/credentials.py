from dotenv import dotenv_values

import streamlit as st

from paths import REPO_ROOT

GLM_KEY_NAME = "glm_api_key"
HUGGINGFACE_KEY_NAME = "huggingface_api_key"


def load_dotenv_values() -> dict:
    """Read a local .env if present; never raises if it's missing."""
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return {}
    return dict(dotenv_values(env_path))


def resolve_initial_key(existing: str | None, dotenv_value: str | None) -> str:
    if existing:
        return existing
    if dotenv_value:
        return dotenv_value
    return ""


def init_credential_state(session_state: dict, env_values: dict | None = None) -> None:
    """Seed session_state key fields exactly once; safe to call every rerun."""
    if env_values is None:
        env_values = load_dotenv_values()

    if GLM_KEY_NAME not in session_state:
        session_state[GLM_KEY_NAME] = resolve_initial_key(
            None, env_values.get("GLM_API_KEY")
        )
    if HUGGINGFACE_KEY_NAME not in session_state:
        session_state[HUGGINGFACE_KEY_NAME] = resolve_initial_key(
            None, env_values.get("HUGGINGFACE_API_KEY")
        )


def get_glm_key(session_state: dict) -> str:
    return session_state.get(GLM_KEY_NAME, "")


def get_huggingface_key(session_state: dict) -> str:
    return session_state.get(HUGGINGFACE_KEY_NAME, "")


def missing_key_message(task_label: str, provider_label: str) -> str:
    return f"Enter your {provider_label} API key in the sidebar to use {task_label}."


def render_key_sidebar() -> None:
    """Streamlit wiring — manual/integration verified only."""
    init_credential_state(st.session_state)

    st.sidebar.text_input(
        "GLM API key (Z.ai)", type="password", key=GLM_KEY_NAME
    )
    st.sidebar.text_input(
        "HuggingFace API key", type="password", key=HUGGINGFACE_KEY_NAME
    )

    glm_status = "✅ GLM connected" if get_glm_key(st.session_state) else "⚠️ GLM key missing"
    hf_status = "✅ HuggingFace connected" if get_huggingface_key(st.session_state) else "⚠️ HuggingFace key missing"
    st.sidebar.caption(glm_status)
    st.sidebar.caption(hf_status)
