from dotenv import dotenv_values

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from paths import REPO_ROOT

GLM_ENV_VAR = "GLM_API_KEY"
HUGGINGFACE_ENV_VAR = "HUGGINGFACE_API_KEY"


def load_dotenv_values() -> dict:
    """Read a local .env if present; never raises if it's missing."""
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return {}
    return dict(dotenv_values(env_path))


def resolve_fixed_key(secrets_value: str | None, dotenv_value: str | None) -> str:
    """Fixed (operator-configured) key resolution: deployed secrets first,
    then local .env. There is no per-user value anymore — one key serves
    every visitor to this app instance."""
    if secrets_value:
        return secrets_value
    if dotenv_value:
        return dotenv_value
    return ""


def read_secret(env_var_name: str) -> str | None:
    """st.secrets.get() doesn't return None gracefully when no secrets.toml
    exists anywhere — it raises StreamlitSecretNotFoundError. That's the
    normal case for local development, where only .env is used."""
    try:
        return st.secrets.get(env_var_name)
    except StreamlitSecretNotFoundError:
        return None


def get_glm_key() -> str:
    return resolve_fixed_key(read_secret(GLM_ENV_VAR), load_dotenv_values().get(GLM_ENV_VAR))


def get_huggingface_key() -> str:
    return resolve_fixed_key(
        read_secret(HUGGINGFACE_ENV_VAR), load_dotenv_values().get(HUGGINGFACE_ENV_VAR)
    )


def missing_keys_message() -> str:
    return (
        "NexChat isn't configured yet — the server is missing its API keys. "
        f"Set {GLM_ENV_VAR} and {HUGGINGFACE_ENV_VAR} in a local .env file "
        "or in Streamlit secrets, then restart."
    )
