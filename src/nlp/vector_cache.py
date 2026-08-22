import hashlib

FINGERPRINT_KEY = "vector_store_fingerprint"
STORE_KEY = "vector_store_cache"


def compute_files_fingerprint(uploaded_files: list) -> str:
    """Stable fingerprint of a set of uploaded files by name + content hash."""
    hasher = hashlib.sha256()
    for uploaded_file in uploaded_files:
        hasher.update(uploaded_file.name.encode("utf-8"))
        hasher.update(uploaded_file.getvalue())
    return hasher.hexdigest()


def get_cached_vector_store(session_state: dict, fingerprint: str):
    """Returns the cached store only if it matches the current fingerprint.

    Session-scoped by design (not st.cache_resource, which is process-global
    in Streamlit and would leak one user's embedding-model/key across
    sessions on a fingerprint collision — spec §4.2).
    """
    if session_state.get(FINGERPRINT_KEY) == fingerprint:
        return session_state.get(STORE_KEY)
    return None


def store_vector_store(session_state: dict, fingerprint: str, vector_store) -> None:
    """Stores the single most-recent vector store for this session, replacing any prior one."""
    session_state[FINGERPRINT_KEY] = fingerprint
    session_state[STORE_KEY] = vector_store
