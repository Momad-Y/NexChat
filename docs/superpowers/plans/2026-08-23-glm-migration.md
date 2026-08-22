# GLM-4.7-Flash Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove Gemini entirely from NexChat and replace it with GLM-4.7-Flash for RAG chat and hosted HuggingFace embeddings for RAG retrieval, so the app runs end-to-end on zero-cost, no-credit-card API keys.

**Architecture:** Swap `ChatGoogleGenerativeAI`/`GoogleGenerativeAIEmbeddings` in `src/nlp/RAG.py` for `ChatOpenAI` (OpenAI-compatible, pointed at Z.ai) and `HuggingFaceEndpointEmbeddings` (hosted, reusing the existing HuggingFace BYOK key). Rename the BYOK credential naming from Gemini to GLM throughout. Gate the Question Answering UI on both keys simultaneously instead of sequentially, and wrap vector-store indexing so an embeddings failure shows a friendly message instead of a crash.

**Tech Stack:** Python 3.11, Streamlit, `langchain-openai` (new), `langchain-huggingface` (re-added), a verified-compatible bump of the `langchain`/`langchain-community`/`langchain-core`/`langchain-text-splitters` family (all still 0.3.x — not the 1.x line that broke `langchain.text_splitter` during the prior migration), `pytest`.

**Spec:** `docs/superpowers/specs/2026-08-23-glm-migration-design.md`

## Global Constraints

- **No `os.environ` key mutation, ever.** Every provider client takes the caller's key as a constructor argument (`api_key=...`, `huggingfacehub_api_token=...`). (Spec §2, inherited from the prior migration's binding constraint.)
- **GLM chat endpoint:** `https://api.z.ai/api/paas/v4/`, model string `glm-4.7-flash`, via `langchain_openai.ChatOpenAI`. (Spec §3.1, §4.1)
- **Embeddings:** `HuggingFaceEndpointEmbeddings(model="BAAI/bge-small-en-v1.5", provider="hf-inference", task="feature-extraction", huggingfacehub_api_token=...)` — `provider` MUST be explicitly `"hf-inference"`, never left unset (leaving it `None` triggers a live "auto" marketplace-resolution call with its own failure mode, found by red-team review). (Spec §3.2, §3.8, §4.1)
- **BYOK stays two keys — GLM + HuggingFace** — renamed from Gemini throughout (`GLM_KEY_NAME`, `get_glm_key`, `GLM_API_KEY` env var, "GLM API key (Z.ai)" sidebar label). (Spec §3.4, §4.2)
- **Vector-cache fingerprint folds in the HuggingFace key, not the GLM key** — embeddings identity depends only on the HuggingFace key, confirmed by tracing the actual call pattern (`llm` is never cached; only `vector_store` is). (Spec §3.6, §4.3)
- **Verified exact dependency pin set** (installed for real, confirmed torch-free, confirmed every `RAG.py` import resolves) — do not substitute newer or unpinned versions without re-verifying, since the newest releases of `langchain-openai`/`langchain-huggingface` require `langchain-core>=1.x`, which breaks `langchain.text_splitter`:
  ```
  langchain==0.3.30
  langchain-community==0.3.31
  langchain-core==0.3.86
  langchain-text-splitters==0.3.11
  langchain-openai==0.3.35
  langchain-huggingface==0.3.1
  ```
- **Missing-key UX shows every missing key at once**, not one at a time (sequential `elif` gating was a red-team finding — first-time users with zero keys should not have to fix one, rerun, and only then discover the second is also missing). (Spec §3.10, §4.4)
- **`create_vector_store` failures must show a friendly `st.error(...)` message**, never an uncaught exception — the one place in the QA flow that lacked this app's otherwise-universal error-handling discipline. (Spec §3.9, §4.4)

---

## File Structure

Modified:
- `src/nlp/RAG.py` — chat + embeddings provider swap (`init_llm_model`, `init_embeddings_model`, `init_RAG`; type annotations on `create_vector_store`/`create_qa_model`)
- `src/credentials.py` — Gemini → GLM naming throughout
- `src/nlp/vector_cache.py` — fingerprint parameter rename (Gemini key → HuggingFace key)
- `src/app.py` — Question Answering branch: simultaneous key gating, updated `init_RAG`/`compute_files_fingerprint` calls, wrapped `create_vector_store`
- `requirements.txt` — regenerated with the verified pin set
- `README.md` — six Gemini references corrected
- `tests/test_rag.py`, `tests/test_credentials.py`, `tests/test_vector_cache.py` — updated for the above

No new files. No files removed.

---

### Task 1: `src/nlp/RAG.py` — swap chat and embeddings providers

**Files:**
- Modify: `src/nlp/RAG.py` (imports; `init_llm_model`, `init_embeddings_model`, `init_RAG`; type annotations on `create_vector_store`, `create_qa_model`)
- Test: `tests/test_rag.py` (replace the two Gemini-specific tests)

**Interfaces:**
- Produces: `init_llm_model(glm_api_key: str) -> ChatOpenAI`, `init_embeddings_model(huggingface_api_key: str) -> HuggingFaceEndpointEmbeddings`, `init_RAG(glm_api_key: str, huggingface_api_key: str) -> tuple` (llm, embedding_model, prompt, contextualize_q_prompt) — the two-argument signature is new and load-bearing for Task 4.
- Consumes: nothing from other tasks. `create_vector_store`, `create_qa_model`, `build_chat_history`, `qa()`, `init_prompt`, `QA_ERROR_MESSAGE`, `QA_INTERRUPTED_SUFFIX`, `is_qa_failure` are all unchanged by this task — do not touch them beyond the two type-annotation edits below.

- [ ] **Step 1: Install the verified dependency set into the dev venv**

```bash
.venv/bin/pip install "langchain==0.3.30" "langchain-community==0.3.31" "langchain-core==0.3.86" "langchain-text-splitters==0.3.11" "langchain-openai==0.3.35" "langchain-huggingface==0.3.1"
```

This upgrades the langchain family in place (from the versions the prior migration pinned) to versions verified compatible with both new packages. `langchain-google-genai` stays installed for now — it's still imported by the current `RAG.py` until Step 3 replaces those imports; it gets removed from the dependency manifest in Task 5, not here.

- [ ] **Step 2: Write the failing tests**

Replace the two existing Gemini-specific tests at the top of `tests/test_rag.py` (`test_init_llm_model_passes_key_as_argument_not_env` and `test_init_embeddings_model_passes_key_as_argument`) with:

```python
import os


def test_init_llm_model_passes_key_as_argument_not_env(monkeypatch):
    from nlp.RAG import init_llm_model

    captured = {}

    class FakeChatModel:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("nlp.RAG.ChatOpenAI", FakeChatModel)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    init_llm_model("test-glm-key")

    assert captured["api_key"] == "test-glm-key"
    assert captured["base_url"] == "https://api.z.ai/api/paas/v4/"
    assert captured["model"] == "glm-4.7-flash"
    assert "OPENAI_API_KEY" not in os.environ


def test_init_embeddings_model_passes_key_as_argument(monkeypatch):
    from nlp.RAG import init_embeddings_model

    captured = {}

    class FakeEmbeddings:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("nlp.RAG.HuggingFaceEndpointEmbeddings", FakeEmbeddings)

    init_embeddings_model("test-hf-key")

    assert captured["huggingfacehub_api_token"] == "test-hf-key"
    assert captured["model"] == "BAAI/bge-small-en-v1.5"
    assert captured["provider"] == "hf-inference"
    assert captured["task"] == "feature-extraction"
```

Every other test in `tests/test_rag.py` (`test_build_chat_history_*`, `test_qa_*`, `test_is_qa_failure_*`) is unchanged — leave them exactly as they are.

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/pytest tests/test_rag.py -v -k "init_llm_model or init_embeddings_model"`
Expected: FAIL — `nlp.RAG.ChatOpenAI`/`nlp.RAG.HuggingFaceEndpointEmbeddings` don't exist yet (the current file still imports `ChatGoogleGenerativeAI`/`GoogleGenerativeAIEmbeddings`), and `init_llm_model`/`init_embeddings_model` still take a `gemini_api_key` argument name mismatched with the new assertions.

- [ ] **Step 4: Implement**

Replace the import block at the top of `src/nlp/RAG.py` (currently `from langchain_google_genai import ChatGoogleGenerativeAI` and `from langchain_google_genai import GoogleGenerativeAIEmbeddings`) with:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import Runnable

from utils import read_file

from streamlit.runtime.uploaded_file_manager import UploadedFile
from typing import Generator
```

Replace `init_llm_model` and `init_embeddings_model` with:

```python
GLM_BASE_URL = "https://api.z.ai/api/paas/v4/"
GLM_MODEL = "glm-4.7-flash"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def init_llm_model(glm_api_key: str) -> ChatOpenAI:
    """
    Initializes the GLM-4.7-Flash chat model via its OpenAI-compatible API.

    Args:
        glm_api_key (str): The caller's GLM (Z.ai) API key.

    Returns:
        ChatOpenAI: An instance configured against Z.ai's GLM endpoint.
    """
    return ChatOpenAI(
        model=GLM_MODEL,
        api_key=glm_api_key,
        base_url=GLM_BASE_URL,
        temperature=0.1,
        max_retries=2,
        timeout=60,
    )


def init_embeddings_model(huggingface_api_key: str) -> HuggingFaceEndpointEmbeddings:
    """
    Initializes hosted HuggingFace embeddings — no local model, no torch.

    Args:
        huggingface_api_key (str): The caller's HuggingFace API key.

    Returns:
        HuggingFaceEndpointEmbeddings: The hosted embeddings model.
    """
    return HuggingFaceEndpointEmbeddings(
        model=EMBEDDING_MODEL,
        provider="hf-inference",
        task="feature-extraction",
        huggingfacehub_api_token=huggingface_api_key,
    )
```

Replace `init_RAG` (leave `init_prompt` between these two functions untouched):

```python
def init_RAG(glm_api_key: str, huggingface_api_key: str) -> tuple:
    """
    Initializes the models and templates for a given user's GLM and HuggingFace keys.

    Args:
        glm_api_key (str): The caller's GLM (Z.ai) API key.
        huggingface_api_key (str): The caller's HuggingFace API key.

    Returns:
        tuple: A tuple of the initialized models and prompt templates.
    """
    llm = init_llm_model(glm_api_key)
    embedding_model = init_embeddings_model(huggingface_api_key)
    prompt, contextualize_q_prompt = init_prompt()

    return llm, embedding_model, prompt, contextualize_q_prompt
```

Update the type annotations (only — the function bodies are unchanged) on `create_vector_store` and `create_qa_model`:

```python
def create_vector_store(
    uploaded_files: list[UploadedFile], embedding_model: HuggingFaceEndpointEmbeddings
) -> FAISS:
```

```python
def create_qa_model(
    vector_store: FAISS,
    llm: ChatOpenAI,
    prompt: ChatPromptTemplate,
    contextualize_q_prompt: ChatPromptTemplate,
) -> Runnable:
```

- [ ] **Step 5: Run to verify pass**

Run: `.venv/bin/pytest tests/test_rag.py -v`
Expected: all tests in the file PASS (the two replaced ones plus every unchanged one — `build_chat_history`, `qa()`, `is_qa_failure` tests are unaffected by this diff and must still pass).

- [ ] **Step 6: Manual pre-implementation verification (requires a real GLM key and a real HuggingFace key — not available in an automated environment)**

```bash
python3 - <<'EOF'
import requests, os
key = os.environ["GLM_API_KEY"]
headers = {"Authorization": f"Bearer {key}"}
r = requests.post(
    "https://api.z.ai/api/paas/v4/chat/completions",
    headers=headers,
    json={"model": "glm-4.7-flash", "messages": [{"role": "user", "content": "Say OK."}]},
)
print(r.status_code, r.json())
EOF
```

Expected: HTTP 200 with a real completion. If `glm-4.7-flash` isn't a valid model string, check Z.ai's current model list for the closest free-tier equivalent and update `GLM_MODEL` in `RAG.py` before proceeding further (spec §5).

Separately, verify the embeddings model resolves on the live HF router:

```bash
python3 - <<'EOF'
from langchain_huggingface import HuggingFaceEndpointEmbeddings
import os
emb = HuggingFaceEndpointEmbeddings(
    model="BAAI/bge-small-en-v1.5",
    provider="hf-inference",
    task="feature-extraction",
    huggingfacehub_api_token=os.environ["HUGGINGFACE_API_KEY"],
)
vector = emb.embed_query("test sentence")
print(len(vector), vector[:5])
EOF
```

Expected: a real vector printed, no exception. If this fails, fall back to `sentence-transformers/all-MiniLM-L6-v2` (spec §3.3) — update `EMBEDDING_MODEL` in `RAG.py`.

If no real keys are available at implementation time, skip this step explicitly (do not fabricate a result) and carry it forward as an outstanding manual verification, the same way the prior migration's HuggingFace endpoint check was handled.

- [ ] **Step 7: Commit**

```bash
git add src/nlp/RAG.py tests/test_rag.py
git commit -m "feat: replace Gemini with GLM-4.7-Flash chat and hosted HuggingFace embeddings"
```

---

### Task 2: `src/credentials.py` — rename Gemini to GLM

**Files:**
- Modify: `src/credentials.py`
- Test: `tests/test_credentials.py`

**Interfaces:**
- Produces: `GLM_KEY_NAME = "glm_api_key"`, `get_glm_key(session_state: dict) -> str` — replaces `GEMINI_KEY_NAME`/`get_gemini_key`. Load-bearing for Task 4.
- Consumes: nothing from other tasks.

- [ ] **Step 1: Write the failing tests**

Update `tests/test_credentials.py`'s two `init_credential_state` tests (change the literal key names, not the test logic):

```python
def test_init_credential_state_seeds_missing_keys_only():
    session_state = {"glm_api_key": "already-set"}
    init_credential_state(session_state, env_values={"HUGGINGFACE_API_KEY": "from-env"})
    assert session_state["glm_api_key"] == "already-set"
    assert session_state["huggingface_api_key"] == "from-env"


def test_init_credential_state_defaults_to_empty_without_dotenv():
    session_state = {}
    init_credential_state(session_state, env_values={})
    assert session_state["glm_api_key"] == ""
    assert session_state["huggingface_api_key"] == ""
```

Update `test_load_dotenv_values_reads_from_repo_root`:

```python
def test_load_dotenv_values_reads_from_repo_root(monkeypatch, tmp_path):
    import credentials

    fake_env = tmp_path / ".env"
    fake_env.write_text("GLM_API_KEY=from-repo-root-env\n")
    monkeypatch.setattr(credentials, "REPO_ROOT", tmp_path)

    values = credentials.load_dotenv_values()

    assert values.get("GLM_API_KEY") == "from-repo-root-env"
```

`test_load_dotenv_values_returns_empty_when_no_env_file`, `test_resolve_initial_key_*`, and `test_missing_key_message_names_task_and_provider` are unchanged — leave them exactly as they are.

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_credentials.py -v`
Expected: the three updated tests FAIL (the current code still reads/writes `gemini_api_key`/`GEMINI_API_KEY`).

- [ ] **Step 3: Implement**

Replace `src/credentials.py` in full:

```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_credentials.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/credentials.py tests/test_credentials.py
git commit -m "feat: rename BYOK Gemini key to GLM throughout credentials.py"
```

---

### Task 3: `src/nlp/vector_cache.py` — fingerprint keys on the HuggingFace key, not the chat key

**Files:**
- Modify: `src/nlp/vector_cache.py`
- Test: `tests/test_vector_cache.py`

**Interfaces:**
- Produces: `compute_files_fingerprint(uploaded_files: list, huggingface_api_key: str) -> str` — parameter renamed from `gemini_api_key`. Load-bearing for Task 4. `get_cached_vector_store`/`store_vector_store` are unchanged.
- Consumes: nothing from other tasks.

- [ ] **Step 1: Update the tests**

In `tests/test_vector_cache.py`, every call to `compute_files_fingerprint(files, "test-gemini-key")` (and the `"key-a"`/`"key-b"`/`"same-key"` calls, which are unaffected either way) stays behaviorally identical — this is a pure parameter rename, so the existing tests already pass unchanged once the implementation is renamed. For clarity matching the new naming, rename the literal string `"test-gemini-key"` to `"test-huggingface-key"` everywhere it appears (four call sites: `test_fingerprint_is_stable_for_identical_files`, `test_fingerprint_changes_when_content_changes`, `test_fingerprint_changes_when_file_set_changes`, `test_fingerprint_distinguishes_ambiguous_name_content_boundary`). This is a cosmetic rename with no behavior change — the string value itself is arbitrary test data.

- [ ] **Step 2: Run to verify current state**

Run: `.venv/bin/pytest tests/test_vector_cache.py -v`
Expected: all tests PASS already (renaming a string literal that's arbitrary test data doesn't change behavior) — this step just confirms the rename didn't typo anything.

- [ ] **Step 3: Implement**

In `src/nlp/vector_cache.py`, rename the parameter (the length-prefixed hashing body is otherwise unchanged):

```python
def compute_files_fingerprint(uploaded_files: list, huggingface_api_key: str) -> str:
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
    key_bytes = huggingface_api_key.encode("utf-8")
    hasher.update(len(key_bytes).to_bytes(8, "big"))
    hasher.update(key_bytes)
    return hasher.hexdigest()
```

`get_cached_vector_store` and `store_vector_store` are untouched.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_vector_cache.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/nlp/vector_cache.py tests/test_vector_cache.py
git commit -m "fix: vector-cache fingerprint keys on the HuggingFace key, not the chat key"
```

---

### Task 4: `src/app.py` — wire GLM + HuggingFace into the Question Answering branch

**Files:**
- Modify: `src/app.py`

**Interfaces:**
- Consumes: `credentials.get_glm_key` (Task 2), `nlp.RAG.init_RAG(glm_api_key, huggingface_api_key)` (Task 1), `nlp.vector_cache.compute_files_fingerprint(uploaded_files, huggingface_api_key)` (Task 3).

This is integration/UI wiring — `src/app.py` has no unit tests in this codebase's established pattern (it's a Streamlit script, verified via manual/`AppTest` checks, consistent with how the prior migration's equivalent app.py-wiring task was verified). No new permanent test file is added here.

- [ ] **Step 1: Update the import line**

Find:
```python
from credentials import render_key_sidebar, get_gemini_key, get_huggingface_key, missing_key_message
```
Replace with:
```python
from credentials import render_key_sidebar, get_glm_key, get_huggingface_key, missing_key_message
```

- [ ] **Step 2: Replace the Question Answering branch's key-gating header**

Find (the start of the `else:` branch for Question Answering, through the `init_RAG` call and the `if uploaded_files:` line):

```python
else:
    gemini_key = get_gemini_key(st.session_state)
    if not gemini_key:
        st.info(missing_key_message("Question Answering", "Gemini"))
    else:
        llm, embedding_model, prompt_template, contextualize_q_prompt = init_RAG(gemini_key)

        uploaded_files = st.file_uploader(
            "Upload a file", type=["pdf", "csv", "txt", "md"], accept_multiple_files=True
        )
        st.divider()

        vector_store = qa_model = None

        if uploaded_files:
            fingerprint = compute_files_fingerprint(uploaded_files, gemini_key)
            vector_store = get_cached_vector_store(st.session_state, fingerprint)
            if vector_store is None:
                with st.spinner("Indexing files…"):
                    vector_store = create_vector_store(uploaded_files, embedding_model)
                    store_vector_store(st.session_state, fingerprint, vector_store)
            else:
                st.caption("Using cached index for this file set.")

            qa_model = create_qa_model(vector_store, llm, prompt_template, contextualize_q_prompt)
```

Replace with:

```python
else:
    glm_key = get_glm_key(st.session_state)
    hf_key = get_huggingface_key(st.session_state)

    if not glm_key:
        st.info(missing_key_message("Question Answering", "GLM"))
    if not hf_key:
        st.info(missing_key_message("Question Answering", "HuggingFace"))

    if glm_key and hf_key:
        llm, embedding_model, prompt_template, contextualize_q_prompt = init_RAG(glm_key, hf_key)

        uploaded_files = st.file_uploader(
            "Upload a file", type=["pdf", "csv", "txt", "md"], accept_multiple_files=True
        )
        st.divider()

        vector_store = qa_model = None

        if uploaded_files:
            fingerprint = compute_files_fingerprint(uploaded_files, hf_key)
            vector_store = get_cached_vector_store(st.session_state, fingerprint)
            if vector_store is None:
                with st.spinner("Indexing files…"):
                    try:
                        vector_store = create_vector_store(uploaded_files, embedding_model)
                        store_vector_store(st.session_state, fingerprint, vector_store)
                    except Exception:
                        st.error("Couldn't index the uploaded files — check your HuggingFace key and try again.")
                        vector_store = None
            else:
                st.caption("Using cached index for this file set.")

            if vector_store is not None:
                qa_model = create_qa_model(vector_store, llm, prompt_template, contextualize_q_prompt)
```

**Everything below this point in the file — the chat-message display loop, the `st.chat_input()` handler, the "Start New Chat" button, the voice-question widget, and the "Audio Output" button — is UNCHANGED.** It was already indented one level under the old inner `else:`, and the new `if glm_key and hf_key:` sits at exactly the same indentation depth as that old `else:` did, so no re-indentation of the rest of the branch is needed — only the header shown above changes. Do not touch anything from the `# Display the chat interface` comment onward.

- [ ] **Step 3: Static verification**

```bash
.venv/bin/python -c "import ast; ast.parse(open('src/app.py').read())"
```

Expected: no output (valid syntax). Then run the full test suite to confirm nothing elsewhere broke on import:

```bash
.venv/bin/pytest tests/ -v
```

Expected: all tests pass (this task adds no new tests but must not regress any existing ones).

- [ ] **Step 4: Manual verification — simultaneous missing-key messages**

```bash
.venv/bin/python - <<'EOF'
from streamlit.testing.v1 import AppTest

at = AppTest.from_file("src/app.py")
at.run()
at.sidebar.selectbox[0].select("Question Answering").run()

infos = [el.value for el in at.info]
assert any("GLM" in i for i in infos), f"no GLM message in {infos}"
assert any("HuggingFace" in i for i in infos), f"no HuggingFace message in {infos}"
assert not at.exception
print("PASS: both missing-key messages shown at once, no crash")
EOF
```

Expected: `PASS: both missing-key messages shown at once, no crash` — confirms §3.10's fix actually works, not just that it was written correctly.

- [ ] **Step 5: Manual verification checklist (requires real GLM + HuggingFace keys — skip explicitly if unavailable, carry forward as outstanding)**

Run: `streamlit run src/app.py`

1. Enter a real GLM key only: Question Answering shows only the HuggingFace-missing message (GLM message gone).
2. Enter both real keys: upload a file, confirm it indexes (spinner, then chat works) and a real answer streams in.
3. Change the GLM key to an obviously invalid string mid-session, ask another question: confirm the error/interrupted message appears in the chat but is NOT replayed as fake history on the next question (§3.6's fingerprint fix means changing the GLM key alone doesn't force re-indexing, which is correct — only the HF key affects the cache; the LLM call itself will simply fail cleanly with the bad key).
4. Revert to a valid GLM key, confirm the conversation continues normally.

- [ ] **Step 6: Commit**

```bash
git add src/app.py
git commit -m "feat: wire GLM + HuggingFace keys into the Question Answering branch"
```

---

### Task 5: Dependencies — regenerate `requirements.txt`

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Regenerate from a clean venv using the verified pin set**

```bash
python3 -m venv /tmp/glm-clean-venv
/tmp/glm-clean-venv/bin/pip install --upgrade pip
/tmp/glm-clean-venv/bin/pip install \
  "langchain==0.3.30" "langchain-community==0.3.31" "langchain-core==0.3.86" \
  "langchain-text-splitters==0.3.11" "langchain-openai==0.3.35" "langchain-huggingface==0.3.1" \
  streamlit faiss-cpu PyPDF2 pandas requests python-dotenv SpeechRecognition gTTS
```

This list is the same direct-dependency set the prior migration's Task 14 verified, with `langchain-google-genai` removed and `langchain-openai`/`langchain-huggingface` added at the pins verified in Task 1 of this plan.

- [ ] **Step 2: Verify torch absence and the real RAG.py import set**

```bash
/tmp/glm-clean-venv/bin/python -c "import torch" 2>&1
```
Expected: `ModuleNotFoundError: No module named 'torch'`

```bash
/tmp/glm-clean-venv/bin/python -c "
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpointEmbeddings
print('OK')
"
```
Expected: `OK`

- [ ] **Step 3: Verify zero Gemini/Google references remain**

```bash
grep -iE "^langchain-google-genai|^google-ai-generativelanguage|^google-api-core|^google-api-python-client|^google-auth|^google-generativeai|^googleapis-common-protos" requirements.txt
```

Note: run this against the OLD `requirements.txt` before Step 4 overwrites it, to confirm you know what you're removing; after Step 4, re-run it against the new file and expect no output.

- [ ] **Step 4: Freeze and write requirements.txt**

```bash
cd /home/momad/Projects/NexChat
/tmp/glm-clean-venv/bin/pip install -q pytest
/tmp/glm-clean-venv/bin/pip freeze | grep -v "^pytest\|^iniconfig\|^pluggy\|^Pygments" > requirements.txt
rm -rf /tmp/glm-clean-venv
```

(The `grep -v` exclusions match the prior migration's Task 14 fix — pytest and its own transitive deps must not leak into the production manifest; installing pytest only to run the verification suite in Step 5, then excluding it from the freeze, reproduces that exact bug class if skipped.)

- [ ] **Step 5: Full-suite verification in a second clean venv**

```bash
python3 -m venv /tmp/glm-verify-venv
/tmp/glm-verify-venv/bin/pip install -r requirements-dev.txt
/tmp/glm-verify-venv/bin/python -m pytest tests/ -v
/tmp/glm-verify-venv/bin/python -c "import torch" 2>&1
rm -rf /tmp/glm-verify-venv
```

Expected: full test suite passes (all tests from Tasks 1-3 plus every pre-existing test), and the `torch` import still fails with `ModuleNotFoundError`.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt
git commit -m "chore: swap langchain-google-genai for langchain-openai + langchain-huggingface"
```

---

### Task 6: `README.md` — correct the six remaining Gemini references

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the Question Answering task description**

Find:
```
-   **Question Answering:** Powered by RAG using `gemini-2.5-flash` as the LLM, hosted Gemini embeddings (`models/text-embedding-004`) as the embedding model — no local embedding model, no torch dependency — and Faiss as the vector store.
```
Replace with:
```
-   **Question Answering:** Powered by RAG using `glm-4.7-flash` (Z.ai) as the LLM, hosted HuggingFace embeddings (`BAAI/bge-small-en-v1.5`) as the embedding model — no local embedding model, no torch dependency — and Faiss as the vector store.
```

- [ ] **Step 2: Update the tech-stack list**

Find:
```
    -   Google Gemini (chat + hosted embeddings) via `langchain-google-genai`
    -   HuggingFace Inference API (hosted) for summarization and image captioning
```
Replace with:
```
    -   GLM-4.7-Flash (Z.ai, OpenAI-compatible) for chat via `langchain-openai`
    -   HuggingFace Inference API (hosted) for embeddings, summarization, and image captioning
```

- [ ] **Step 3: Update the BYOK configuration section**

Find:
```
NexChat is bring-your-own-key. There is nothing to configure before running it:
enter your **Gemini** and **HuggingFace** API keys directly in the app's sidebar at
```
Replace with:
```
NexChat is bring-your-own-key. There is nothing to configure before running it:
enter your **GLM (Z.ai)** and **HuggingFace** API keys directly in the app's sidebar at
```

- [ ] **Step 4: Update the `.env` example**

Find:
```
    GEMINI_API_KEY=your_key_here
    HUGGINGFACE_API_KEY=your_key_here
```
Replace with:
```
    GLM_API_KEY=your_key_here
    HUGGINGFACE_API_KEY=your_key_here
```

- [ ] **Step 5: Update the running-the-application instructions**

Find:
```
2. Enter your Gemini and HuggingFace API keys in the sidebar.
```
Replace with:
```
2. Enter your GLM (Z.ai) and HuggingFace API keys in the sidebar.
```

- [ ] **Step 6: Update the Used Resources list**

Find:
```
-   **Google Gemini API Documentation:** [https://ai.google.dev/gemini-api/docs](https://ai.google.dev/gemini-api/docs)
```
Replace with:
```
-   **Z.ai GLM API Documentation:** [https://docs.z.ai/guides/overview/quick-start](https://docs.z.ai/guides/overview/quick-start)
```

- [ ] **Step 7: Verify no Gemini references remain**

```bash
grep -in gemini README.md
```
Expected: no output.

- [ ] **Step 8: Commit**

```bash
git add README.md
git commit -m "docs: replace Gemini references with GLM throughout README"
```

---

## Self-Review Notes

- **Spec coverage:** every numbered spec decision (§3.1–§3.10) maps to a task above — chat/embeddings provider choice and code (§3.1, §3.2 → Task 1), embedding model (§3.3 → Task 1 Step 6), BYOK naming (§3.4 → Task 2), `init_RAG` signature (§3.5 → Task 1), fingerprint key (§3.6 → Task 3), dependencies (§3.7 → Task 5, with the additional real version-conflict finding folded in), `provider="hf-inference"` (§3.8 → Task 1), `create_vector_store` error handling (§3.9 → Task 4), simultaneous key messaging (§3.10 → Task 4). The testing plan's rate-limit risk note (§5) has no corresponding task — it's explicitly documented as an accepted risk with no new mechanism, not a gap.
- **Placeholder scan:** no TBD/TODO; every step has real code, not a description of code.
- **Type consistency:** `init_RAG(glm_api_key, huggingface_api_key)` (Task 1) matches its call site in Task 4 exactly (`init_RAG(glm_key, hf_key)`, positional, same order). `compute_files_fingerprint(uploaded_files, huggingface_api_key)` (Task 3) matches its call site in Task 4 (`compute_files_fingerprint(uploaded_files, hf_key)`). `get_glm_key(session_state)` (Task 2) matches its import and call in Task 4. `create_vector_store`/`create_qa_model`'s updated type annotations (Task 1) are annotation-only — their call sites in Task 4 are unchanged positional calls, so no signature mismatch is possible.
- **Dependency-version risk, called out explicitly rather than left implicit:** Task 1 Step 1 and Task 5 both use the exact same verified pin set (`langchain==0.3.30`, `langchain-community==0.3.31`, `langchain-core==0.3.86`, `langchain-text-splitters==0.3.11`, `langchain-openai==0.3.35`, `langchain-huggingface==0.3.1`) — this was independently installed and verified for real (not just dry-run) during plan-writing, including confirming `langchain.text_splitter` still imports (the exact line that broke under langchain 1.x during the prior migration). If a future implementer needs to bump any of these, they must re-verify the full family together — the langsmith-ceiling conflict this plan traces is not visible from any single package's own changelog.
