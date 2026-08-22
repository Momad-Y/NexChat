import hashlib

FINGERPRINT_KEY = "vector_store_fingerprint"
STORE_KEY = "vector_store_cache"


def compute_files_fingerprint(uploaded_files: list, gemini_api_key: str) -> str:
    """Stable fingerprint of a set of uploaded files AND the embedding key
    used to index them. Including the key is required so a mid-session key
    change invalidates the cache — otherwise retrieval would silently keep
    using embeddings built with a stale/revoked key."""
    hasher = hashlib.sha256()
    for uploaded_file in uploaded_files:
        name_bytes = uploaded_file.name.encode("utf-8")
        content_bytes = uploaded_file.getvalue()
        hasher.update(len(name_bytes).to_bytes(8, "big"))
        hasher.update(name_bytes)
        hasher.update(len(content_bytes).to_bytes(8, "big"))
        hasher.update(content_bytes)
    key_bytes = gemini_api_key.encode("utf-8")
    hasher.update(len(key_bytes).to_bytes(8, "big"))
    hasher.update(key_bytes)
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
