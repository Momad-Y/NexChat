# NexChat Cloud + BYOK Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make NexChat deploy and run correctly on Streamlit Community Cloud with per-user BYOK credentials, while remaining fully working locally on any OS, with no feature dropped and no dead dependency left behind.

**Architecture:** Replace import-time credential reads with lazy, per-call client construction fed by session-scoped BYOK keys; move RAG embeddings to a hosted Gemini call to drop the torch dependency; rebuild both audio features around browser widgets (`st.audio_input`/`st.audio`) instead of server-side devices; re-point HuggingFace calls at the live Inference Providers router; replace fake buffered "streaming" with real incremental output; fix the chat-history and silent-exception bugs found in the audit.

**Tech Stack:** Python 3.11, Streamlit 1.41, LangChain (`langchain-google-genai`, `langchain-community` FAISS), `google-generativeai` via LangChain, HuggingFace Inference Providers (`requests`), `SpeechRecognition`, `gTTS`, `pytest` (new, dev-only).

**Spec:** `docs/superpowers/specs/2026-08-22-nexchat-cloud-byok-design.md`

## Global Constraints

- **No `os.environ` key mutation, ever.** Every provider client takes the caller's key as a constructor/parameter argument (`google_api_key=...`, per-call `Authorization` header). Community Cloud can serve multiple sessions from one shared process; env mutation is a cross-user key leak. (Spec §4.1)
- **No `st.cache_resource` for anything holding a user's key or a client built from it.** `st.cache_resource` is process-global in Streamlit; caching must be `st.session_state`-scoped instead. (Spec §4.2)
- **BYOK keys are session-only.** Read from `st.session_state["gemini_api_key"]` / `st.session_state["huggingface_api_key"]`, seeded from a local `.env` only as a dev-convenience prefill, never written to disk, never logged. (Spec §3)
- **HuggingFace endpoint:** `https://router.huggingface.co/hf-inference/models/{model}` (legacy `api-inference.huggingface.co` is dead — confirmed via `curl`). Summarization model: `facebook/bart-large-cnn`. Captioning model: `Salesforce/blip-image-captioning-base`. (Spec §4.3)
- **Embeddings:** `GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=...)` — no local model, no torch. (Spec §4.2)
- **Audio is entirely in-browser:** `st.audio_input` for capture, `st.audio` for playback. No PyAudio, no pygame, no chime, no server-side device access. (Spec §4.4)
- **`requirements.txt` is regenerated from a clean venv install + `pip freeze`**, not hand-edited, to avoid stranding transitive deps. (Spec §4.8)

---

## File Structure

New:
- `src/paths.py` — repo-relative asset path resolution
- `src/credentials.py` — BYOK key state (pure resolve logic + thin sidebar renderer)
- `src/nlp/vector_cache.py` — session-scoped vector store fingerprint/cache
- `requirements-dev.txt` — `pytest`, dev-only, never installed on Cloud
- `conftest.py` — adds `src/` to `sys.path` for tests, matching the app's existing flat-import style
- `tests/test_paths.py`, `tests/test_credentials.py`, `tests/test_vector_cache.py`, `tests/test_rag.py`, `tests/test_summarization.py`, `tests/test_image_captioning.py`, `tests/test_audio_input.py`, `tests/test_audio_output.py`

Modified:
- `src/app.py` — credential gating, path fixes, streaming control flow, audio widget wiring
- `src/nlp/RAG.py` — lazy factories, chat-history fix, real streaming + mid-stream error handling
- `src/nlp/summarization.py` — lazy factory, endpoint migration, streaming error handling
- `src/cv/image_captioning.py` — lazy factory, endpoint migration
- `src/audio/audio_input.py` — `sr.AudioFile`-based recognition, idempotency guard, malformed-blob handling
- `src/audio/audio_output.py` — returns MP3 bytes instead of playing them
- `requirements.txt`, `packages.txt`, `.devcontainer/devcontainer.json` — dependency slimming

Removed:
- `test/test.py` — a manual print-script against the dead legacy endpoint, not an automated test; superseded by `tests/`

---

### Task 1: Test scaffolding

**Files:**
- Create: `requirements-dev.txt`
- Create: `conftest.py`
- Create: `tests/__init__.py` (empty)
- Create: `tests/test_smoke.py`
- Delete: `test/test.py`

**Interfaces:**
- Produces: a working `pytest` invocation from the repo root that can `from utils import ...`, `from nlp.RAG import ...`, etc., matching the app's existing flat-src import style.

- [ ] **Step 1: Create the dev requirements file**

```
# requirements-dev.txt
-r requirements.txt
pytest==8.3.4
```

- [ ] **Step 2: Create conftest.py at repo root**

```python
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
```

- [ ] **Step 3: Create empty tests package marker**

```bash
mkdir -p tests
touch tests/__init__.py
```

- [ ] **Step 4: Write a smoke test**

```python
# tests/test_smoke.py
def test_pytest_can_import_src():
    from utils import get_file_extension

    assert get_file_extension("report.pdf") == "pdf"
```

- [ ] **Step 5: Install dev requirements and run**

Run: `pip install -r requirements-dev.txt && pytest tests/test_smoke.py -v`
Expected: PASS

- [ ] **Step 6: Remove the stray manual test script**

```bash
git rm test/test.py
```

- [ ] **Step 7: Commit**

```bash
git add requirements-dev.txt conftest.py tests/__init__.py tests/test_smoke.py
git commit -m "test: add pytest scaffolding, remove stray manual test script"
```

---

### Task 2: Asset path resolution

**Files:**
- Create: `src/paths.py`
- Test: `tests/test_paths.py`
- Modify: `src/app.py:37,47`

**Interfaces:**
- Produces: `asset_path(filename: str) -> Path` — resolves a filename inside the repo's `imgs/` directory regardless of process CWD.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_paths.py
def test_asset_path_resolves_existing_icon():
    from paths import asset_path

    resolved = asset_path("icon.png")
    assert resolved.name == "icon.png"
    assert resolved.exists()


def test_asset_path_resolves_regardless_of_cwd(tmp_path, monkeypatch):
    from paths import asset_path

    monkeypatch.chdir(tmp_path)
    resolved = asset_path("logo.png")
    assert resolved.exists()
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_paths.py -v`
Expected: FAIL with "No module named 'paths'"

- [ ] **Step 3: Implement**

```python
# src/paths.py
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "imgs"


def asset_path(filename: str) -> Path:
    """Resolve a filename inside imgs/, independent of process CWD."""
    return ASSETS_DIR / filename
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_paths.py -v`
Expected: PASS

- [ ] **Step 5: Wire into app.py**

In `src/app.py`, replace the CWD-relative literals:

```python
# before
st.set_page_config(
    page_title="NexChat",
    page_icon="./imgs/icon.png",
    layout="centered",
    initial_sidebar_state="expanded",
)
...
st.sidebar.image("./imgs/logo.png", use_container_width=True)
```

```python
# after
from paths import asset_path

st.set_page_config(
    page_title="NexChat",
    page_icon=str(asset_path("icon.png")),
    layout="centered",
    initial_sidebar_state="expanded",
)
...
st.sidebar.image(str(asset_path("logo.png")), use_container_width=True)
```

- [ ] **Step 6: Manually verify both launch conventions**

Run from repo root: `streamlit run src/app.py --server.headless true &` then `curl -sf http://localhost:8501 >/dev/null && echo OK`; stop it.
Run from `src/`: `cd src && streamlit run app.py --server.headless true &` then repeat the curl check; stop it.
Expected: both start without a missing-image error in the terminal log.

- [ ] **Step 7: Commit**

```bash
git add src/paths.py tests/test_paths.py src/app.py
git commit -m "fix: resolve asset paths relative to repo root, not CWD"
```

---

### Task 3: BYOK credential state

**Files:**
- Create: `src/credentials.py`
- Test: `tests/test_credentials.py`

**Interfaces:**
- Produces:
  - `load_dotenv_values() -> dict` — `{}` if no `.env` file exists.
  - `resolve_initial_key(existing: str | None, dotenv_value: str | None) -> str`
  - `init_credential_state(session_state: dict) -> None` — seeds `session_state["gemini_api_key"]` / `["huggingface_api_key"]` if absent.
  - `get_gemini_key(session_state: dict) -> str`
  - `get_huggingface_key(session_state: dict) -> str`
  - `missing_key_message(task_label: str, provider_label: str) -> str`
  - `render_key_sidebar() -> None` — thin Streamlit wiring; not unit tested (documented manual check in Task 4).
- Consumes: nothing from earlier tasks.

- [ ] **Step 1: Write the failing tests for pure logic**

```python
# tests/test_credentials.py
from credentials import resolve_initial_key, missing_key_message, init_credential_state


def test_resolve_initial_key_prefers_existing_session_value():
    assert resolve_initial_key("session-key", "dotenv-key") == "session-key"


def test_resolve_initial_key_falls_back_to_dotenv():
    assert resolve_initial_key(None, "dotenv-key") == "dotenv-key"


def test_resolve_initial_key_falls_back_to_empty_string():
    assert resolve_initial_key(None, None) == ""


def test_missing_key_message_names_task_and_provider():
    msg = missing_key_message("Image Captioning", "HuggingFace")
    assert "HuggingFace" in msg
    assert "Image Captioning" in msg


def test_init_credential_state_seeds_missing_keys_only():
    session_state = {"gemini_api_key": "already-set"}
    init_credential_state(session_state, dotenv_values={"HUGGINGFACE_API_KEY": "from-env"})
    assert session_state["gemini_api_key"] == "already-set"
    assert session_state["huggingface_api_key"] == "from-env"


def test_init_credential_state_defaults_to_empty_without_dotenv():
    session_state = {}
    init_credential_state(session_state, dotenv_values={})
    assert session_state["gemini_api_key"] == ""
    assert session_state["huggingface_api_key"] == ""
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_credentials.py -v`
Expected: FAIL with "No module named 'credentials'"

- [ ] **Step 3: Implement**

```python
# src/credentials.py
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
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_credentials.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/credentials.py tests/test_credentials.py
git commit -m "feat: add session-scoped BYOK credential state"
```

---

### Task 4: Wire credentials into the sidebar, fix repo link

**Files:**
- Modify: `src/app.py`

**Interfaces:**
- Consumes: `credentials.render_key_sidebar()`, `credentials.get_gemini_key(session_state)`, `credentials.get_huggingface_key(session_state)`, `credentials.missing_key_message(task_label, provider_label)`.

- [ ] **Step 1: Replace the static sidebar block**

In `src/app.py`, replace the author/repo links block (`app.py:56-67`) and add the key sidebar call right after `st.sidebar.image(...)`:

```python
from credentials import render_key_sidebar, get_gemini_key, get_huggingface_key, missing_key_message

...

st.sidebar.image(str(asset_path("logo.png")), use_container_width=True)

render_key_sidebar()

with st.sidebar.expander("About"):
    st.markdown("## **[Repository Link](https://github.com/Momad-Y/NexChat)**")
    st.markdown("## Done By:")
    st.markdown("##### **Begad M Tamim**")
    st.markdown(
        "##### [Github](https://github.com/begad-tamim) | [LinkedIn](https://www.linkedin.com/in/begad-tamim/) | [Email](mailto:begadtamim.a@gmail.com)"
    )
    st.markdown("##### **Mohamed Y Abdelnasser**")
    st.markdown(
        "##### [Github](https://github.com/Momad-Y) | [LinkedIn](https://www.linkedin.com/in/mohamed-y-abdelnasser/) | [Email](mailto:Mohamed.Y.Abdelnasser@gmail.com)"
    )
```

- [ ] **Step 2: Manually verify in the browser**

Run: `streamlit run src/app.py`
Expected: sidebar shows logo, two password-masked key fields with status captions, and a collapsed "About" expander with a working GitHub link — no crash with empty keys.

- [ ] **Step 3: Commit**

```bash
git add src/app.py
git commit -m "feat: wire BYOK sidebar, fix stale GitLab link"
```

---

### Task 5: RAG.py — lazy factories, drop import-time key reads

**Files:**
- Modify: `src/nlp/RAG.py:1-30,32-50,53-66,121-135`
- Test: `tests/test_rag.py`

**Interfaces:**
- Produces:
  - `init_llm_model(gemini_api_key: str) -> ChatGoogleGenerativeAI`
  - `init_embeddings_model(gemini_api_key: str) -> GoogleGenerativeAIEmbeddings`
  - `init_RAG(gemini_api_key: str) -> tuple` (llm, embedding_model, prompt, contextualize_q_prompt)
- Consumes: `credentials.get_gemini_key` (via app.py call site, not RAG.py itself).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_rag.py
import os


def test_init_llm_model_passes_key_as_argument_not_env(monkeypatch):
    from nlp.RAG import init_llm_model

    captured = {}

    class FakeChatModel:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("nlp.RAG.ChatGoogleGenerativeAI", FakeChatModel)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    init_llm_model("test-gemini-key")

    assert captured["google_api_key"] == "test-gemini-key"
    assert "GOOGLE_API_KEY" not in os.environ


def test_init_embeddings_model_passes_key_as_argument(monkeypatch):
    from nlp.RAG import init_embeddings_model

    captured = {}

    class FakeEmbeddings:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("nlp.RAG.GoogleGenerativeAIEmbeddings", FakeEmbeddings)

    init_embeddings_model("test-gemini-key")

    assert captured["google_api_key"] == "test-gemini-key"
    assert captured["model"] == "models/text-embedding-004"
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_rag.py -v`
Expected: FAIL (`init_llm_model` still reads `os.environ`, `GoogleGenerativeAIEmbeddings` not yet imported)

- [ ] **Step 3: Implement**

In `src/nlp/RAG.py`, delete the entire top-level `try/except` block (lines 22-29) that sets `os.environ[...]`, and delete the now-unused `HuggingFaceEmbeddings` import, replacing it with `GoogleGenerativeAIEmbeddings`:

```python
# remove:
# from langchain_huggingface import HuggingFaceEmbeddings
# ...
# try:
#     os.environ["HUGGINGFACEHUB_API_TOKEN"] = dotenv_values(find_dotenv())[...]
#     os.environ["GOOGLE_API_KEY"] = dotenv_values(find_dotenv())["GEMINI_API_KEY"]
# except Exception as e:
#     os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACE_API_KEY"]
#     os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]

# add:
from langchain_google_genai import GoogleGenerativeAIEmbeddings
```

```python
def init_llm_model(gemini_api_key: str) -> ChatGoogleGenerativeAI:
    """Initializes the ChatGoogleGenerativeAI model with an explicit key."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=gemini_api_key,
        temperature=0.1,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )


def init_embeddings_model(gemini_api_key: str) -> GoogleGenerativeAIEmbeddings:
    """Initializes hosted Gemini embeddings — no local model, no torch."""
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=gemini_api_key,
    )
```

```python
def init_RAG(gemini_api_key: str) -> tuple:
    """Initializes the models and templates for a given user's Gemini key."""
    llm = init_llm_model(gemini_api_key)
    embedding_model = init_embeddings_model(gemini_api_key)
    prompt, contextualize_q_prompt = init_prompt()

    return llm, embedding_model, prompt, contextualize_q_prompt
```

Also remove the now-dead `dotenv_values`/`find_dotenv` import and the `os` import if nothing else in the file uses `os` (check after Task 9's error-handling changes; leave `import os` if still referenced).

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_rag.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/nlp/RAG.py tests/test_rag.py
git commit -m "feat: RAG.py takes Gemini key as argument, drop torch-backed embeddings"
```

---

### Task 6: summarization.py + image_captioning.py — lazy factories, endpoint migration

**Files:**
- Modify: `src/nlp/summarization.py`
- Modify: `src/cv/image_captioning.py`
- Test: `tests/test_summarization.py`
- Test: `tests/test_image_captioning.py`

**Interfaces:**
- Produces:
  - `summarize_text(text: str, huggingface_api_key: str) -> Generator[str, None, None]`
  - `caption_image(uploaded_file: UploadedFile, huggingface_api_key: str) -> str`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_summarization.py
def test_summarize_text_calls_router_endpoint_with_key(monkeypatch):
    from nlp.summarization import summarize_text

    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return [{"summary_text": "a short summary"}]

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        return FakeResponse()

    monkeypatch.setattr("nlp.summarization.requests.post", fake_post)

    chunks = list(summarize_text("some text to summarize", "test-hf-key"))

    assert captured["url"] == "https://router.huggingface.co/hf-inference/models/facebook/bart-large-cnn"
    assert captured["headers"]["Authorization"] == "Bearer test-hf-key"
    assert "".join(chunks)


def test_summarize_text_yields_error_message_on_request_failure(monkeypatch):
    from nlp.summarization import summarize_text

    def fake_post(*args, **kwargs):
        raise ConnectionError("network down")

    monkeypatch.setattr("nlp.summarization.requests.post", fake_post)

    chunks = list(summarize_text("some text", "test-hf-key"))

    assert chunks == ["An error occurred while generating the summary."]
```

```python
# tests/test_image_captioning.py
def test_caption_image_calls_router_endpoint_with_key(monkeypatch):
    from cv.image_captioning import caption_image

    captured = {}

    class FakeResponse:
        def json(self):
            return [{"generated_text": "a photo of a cat"}]

    class FakeUploadedFile:
        def getvalue(self):
            return b"fake-bytes"

    def fake_post(url, headers=None, data=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        return FakeResponse()

    monkeypatch.setattr("cv.image_captioning.requests.post", fake_post)

    caption = caption_image(FakeUploadedFile(), "test-hf-key")

    assert captured["url"] == "https://router.huggingface.co/hf-inference/models/Salesforce/blip-image-captioning-base"
    assert captured["headers"]["Authorization"] == "Bearer test-hf-key"
    assert caption == "A photo of a cat."
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_summarization.py tests/test_image_captioning.py -v`
Expected: FAIL (both functions don't yet take a key argument, both still hit the legacy URL)

- [ ] **Step 3: Implement summarization.py**

```python
# src/nlp/summarization.py
import requests
from typing import Generator
import time

API_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-cnn"
MAX_CHUNK_SIZE = 1000
MAX_NEW_TOKENS = 50


def summarize_text(text: str, huggingface_api_key: str) -> Generator[str, str, str]:
    """Summarizes text via the HuggingFace Inference Providers router."""
    headers = {"Authorization": f"Bearer {huggingface_api_key}"}

    if len(text) > MAX_CHUNK_SIZE:
        text_chunks = [
            text[i : i + MAX_CHUNK_SIZE] for i in range(0, len(text), MAX_CHUNK_SIZE)
        ]

        for chunk in text_chunks:
            payload = {
                "inputs": chunk,
                "parameters": {"max_new_tokens": MAX_NEW_TOKENS},
            }
            try:
                response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                yield response.json()[0]["summary_text"] + " "
            except Exception:
                yield "An error occurred while generating the summary."
                return

    else:
        payload = {"inputs": text}
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            summary = response.json()[0]["summary_text"]
        except Exception:
            yield "An error occurred while generating the summary."
            return

        summary = "**Summary:** " + summary.capitalize().strip()
        sentences = summary.split(".")
        sentences = [sentence for sentence in sentences if sentence]

        for sentence in sentences:
            yield sentence.strip() + ". "
```

Note: the `time.sleep(0.5)` pacing delay is removed (spec §4.5) — the `import time` line is deleted since nothing else in the file uses it.

- [ ] **Step 4: Implement image_captioning.py**

```python
# src/cv/image_captioning.py
import requests
from streamlit.runtime.uploaded_file_manager import UploadedFile

API_URL = "https://router.huggingface.co/hf-inference/models/Salesforce/blip-image-captioning-base"


def caption_image(uploaded_file: UploadedFile, huggingface_api_key: str) -> str:
    """Captions an image via the HuggingFace Inference Providers router."""
    headers = {"Authorization": f"Bearer {huggingface_api_key}"}
    data = uploaded_file.getvalue()

    try:
        response = requests.post(API_URL, headers=headers, data=data, timeout=30)
        caption = response.json()[0]["generated_text"]
    except Exception as e:
        return f"An error occurred while generating the caption: {e}"

    return caption.capitalize().strip() + "."
```

- [ ] **Step 5: Run to verify pass**

Run: `pytest tests/test_summarization.py tests/test_image_captioning.py -v`
Expected: PASS

- [ ] **Step 6: Manual pre-implementation verification (requires a real HuggingFace key)**

```bash
python3 - <<'EOF'
import requests, os
key = os.environ["HUGGINGFACE_API_KEY"]
headers = {"Authorization": f"Bearer {key}"}
r = requests.post(
    "https://router.huggingface.co/hf-inference/models/facebook/bart-large-cnn",
    headers=headers, json={"inputs": "This is a short test sentence to summarize for verification."},
)
print(r.status_code, r.json())
EOF
```

Expected: HTTP 200 with a `summary_text` field. If this 404s, substitute the closest currently-served BART/T5 summarization model on the same provider (spec §4.3 contingency) and repeat this step before moving on — do not proceed to Task 8's summarization wiring with an unverified model.

Repeat the same check against
`https://router.huggingface.co/hf-inference/models/Salesforce/blip-image-captioning-base`
with `data=<raw image bytes>` instead of `json=...`, expecting a `generated_text` field.

- [ ] **Step 7: Commit**

```bash
git add src/nlp/summarization.py src/cv/image_captioning.py tests/test_summarization.py tests/test_image_captioning.py
git commit -m "fix: migrate HuggingFace calls to the live Inference Providers router, take key as argument"
```

---

### Task 7: Session-scoped vector store cache

**Files:**
- Create: `src/nlp/vector_cache.py`
- Test: `tests/test_vector_cache.py`

**Interfaces:**
- Produces:
  - `compute_files_fingerprint(uploaded_files: list) -> str`
  - `get_cached_vector_store(session_state: dict, fingerprint: str)` — returns the cached store or `None`.
  - `store_vector_store(session_state: dict, fingerprint: str, vector_store) -> None`
- Consumes: nothing from earlier tasks (used by app.py in Task 13, alongside `RAG.create_vector_store`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_vector_cache.py
from nlp.vector_cache import compute_files_fingerprint, get_cached_vector_store, store_vector_store


class FakeUploadedFile:
    def __init__(self, name, content):
        self.name = name
        self._content = content

    def getvalue(self):
        return self._content


def test_fingerprint_is_stable_for_identical_files():
    files_a = [FakeUploadedFile("doc.pdf", b"hello world")]
    files_b = [FakeUploadedFile("doc.pdf", b"hello world")]
    assert compute_files_fingerprint(files_a) == compute_files_fingerprint(files_b)


def test_fingerprint_changes_when_content_changes():
    files_a = [FakeUploadedFile("doc.pdf", b"hello world")]
    files_b = [FakeUploadedFile("doc.pdf", b"different content")]
    assert compute_files_fingerprint(files_a) != compute_files_fingerprint(files_b)


def test_fingerprint_changes_when_file_set_changes():
    one_file = [FakeUploadedFile("doc.pdf", b"hello world")]
    two_files = one_file + [FakeUploadedFile("doc2.pdf", b"more")]
    assert compute_files_fingerprint(one_file) != compute_files_fingerprint(two_files)


def test_cache_roundtrip_within_session():
    session_state = {}
    fingerprint = "abc123"
    sentinel_store = object()

    assert get_cached_vector_store(session_state, fingerprint) is None

    store_vector_store(session_state, fingerprint, sentinel_store)

    assert get_cached_vector_store(session_state, fingerprint) is sentinel_store


def test_cache_miss_when_fingerprint_differs():
    session_state = {}
    store_vector_store(session_state, "fingerprint-a", object())

    assert get_cached_vector_store(session_state, "fingerprint-b") is None
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_vector_cache.py -v`
Expected: FAIL with "No module named 'nlp.vector_cache'"

- [ ] **Step 3: Implement**

```python
# src/nlp/vector_cache.py
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
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_vector_cache.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/nlp/vector_cache.py tests/test_vector_cache.py
git commit -m "feat: session-scoped vector store cache, avoids cross-user key leak"
```

---

### Task 8: RAG.py — fix chat-history accumulation bug

**Files:**
- Modify: `src/nlp/RAG.py:215-250` (current `qa()`; exact lines shift after Task 5's edits — locate by function name)
- Test: `tests/test_rag.py` (extend)

**Interfaces:**
- Produces: `build_chat_history(messages: list[dict]) -> list` — pure function, used by `qa()` (Task 9).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_rag.py (append)
def test_build_chat_history_accumulates_all_turns():
    from nlp.RAG import build_chat_history
    from langchain_core.messages import HumanMessage, AIMessage

    messages = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "second answer"},
    ]

    history = build_chat_history(messages)

    assert len(history) == 4
    assert isinstance(history[0], HumanMessage) and history[0].content == "first question"
    assert isinstance(history[1], AIMessage) and history[1].content == "first answer"
    assert isinstance(history[2], HumanMessage) and history[2].content == "second question"
    assert isinstance(history[3], AIMessage) and history[3].content == "second answer"


def test_build_chat_history_handles_trailing_unanswered_user_message():
    from nlp.RAG import build_chat_history

    messages = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "unanswered question"},
    ]

    history = build_chat_history(messages)

    assert len(history) == 3


def test_build_chat_history_handles_empty_messages():
    from nlp.RAG import build_chat_history

    assert build_chat_history([]) == []
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_rag.py -v -k build_chat_history`
Expected: FAIL with "No module named" or "cannot import name 'build_chat_history'"

- [ ] **Step 3: Implement**

Add to `src/nlp/RAG.py`, replacing the buggy loop inside the old `qa()` (which reassigned `user_message`/`ai_answer` each iteration and only kept the last pair):

```python
from langchain_core.messages import HumanMessage, AIMessage


def build_chat_history(messages: list) -> list:
    """Converts the full session message list into ordered LangChain messages."""
    history = []
    for message in messages:
        if message["role"] == "user":
            history.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            history.append(AIMessage(content=message["content"]))
    return history
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_rag.py -v -k build_chat_history`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/nlp/RAG.py tests/test_rag.py
git commit -m "fix: qa() now sees the full chat history, not just the last turn"
```

---

### Task 9: RAG.py — real streaming with mid-stream error handling

**Files:**
- Modify: `src/nlp/RAG.py` (`qa()`)
- Test: `tests/test_rag.py` (extend)

**Interfaces:**
- Produces: `qa(text: str, qa_model: Runnable, messages: list) -> Generator[str, None, None]` — always yields at least one chunk; never raises.
- Consumes: `build_chat_history` (Task 8).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_rag.py (append)
def test_qa_streams_incremental_chunks():
    from nlp.RAG import qa

    class FakeStreamingModel:
        def stream(self, inputs):
            yield {"answer": "Hello"}
            yield {"answer": " world"}

    chunks = list(qa("a question", FakeStreamingModel(), []))

    assert chunks == ["Hello", " world"]


def test_qa_yields_error_message_when_stream_cannot_start():
    from nlp.RAG import qa

    class FakeBrokenModel:
        def stream(self, inputs):
            raise RuntimeError("invalid key")

    chunks = list(qa("a question", FakeBrokenModel(), []))

    assert chunks == ["An error occurred while generating the answer."]


def test_qa_yields_interruption_message_mid_stream():
    from nlp.RAG import qa

    class FakeInterruptedModel:
        def stream(self, inputs):
            yield {"answer": "Partial"}
            raise RuntimeError("connection dropped")

    chunks = list(qa("a question", FakeInterruptedModel(), []))

    assert chunks[0] == "Partial"
    assert "interrupted" in chunks[-1].lower()


def test_qa_yields_error_message_when_stream_produces_nothing():
    from nlp.RAG import qa

    class FakeEmptyModel:
        def stream(self, inputs):
            return iter([])

    chunks = list(qa("a question", FakeEmptyModel(), []))

    assert chunks == ["An error occurred while generating the answer."]
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_rag.py -v -k test_qa`
Expected: FAIL (current `qa()` uses `.invoke()` and word-by-word `time.sleep`, not `.stream()`)

- [ ] **Step 3: Implement**

Replace the body of `qa()` in `src/nlp/RAG.py`:

```python
def qa(text: str, qa_model: Runnable, messages: list) -> Generator[str, None, None]:
    """Streams the answer incrementally; always yields, never raises."""
    chat_history = build_chat_history(messages)

    try:
        stream = qa_model.stream({"chat_history": chat_history, "input": text})
    except Exception:
        yield "An error occurred while generating the answer."
        return

    yielded_any = False
    try:
        for chunk in stream:
            piece = chunk.get("answer")
            if piece:
                yielded_any = True
                yield piece
    except Exception:
        yield "\n\n⚠️ Response interrupted: could not reach Gemini."
        return

    if not yielded_any:
        yield "An error occurred while generating the answer."
```

Remove the now-unused `time` import if nothing else in the file uses it.

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_rag.py -v -k test_qa`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/nlp/RAG.py tests/test_rag.py
git commit -m "feat: stream QA answers incrementally, handle mid-stream failures gracefully"
```

---

### Task 10: summarization.py — remove artificial pacing, confirm error resilience

**Files:**
- Modify: `src/nlp/summarization.py` (already endpoint-migrated in Task 6)
- Test: `tests/test_summarization.py` (extend)

This task is a check, not new functionality — Task 6 already removed `time.sleep(0.5)` and added per-request `try/except`. This task adds the mid-loop (chunked) failure test that Task 6 didn't cover, since the chunked branch's error handling differs slightly from the single-request branch.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_summarization.py (append)
def test_summarize_text_chunked_path_yields_error_on_failure(monkeypatch):
    from nlp.summarization import summarize_text

    def fake_post(*args, **kwargs):
        raise ConnectionError("network down")

    monkeypatch.setattr("nlp.summarization.requests.post", fake_post)

    long_text = "word " * 400  # exceeds MAX_CHUNK_SIZE, exercises the chunked branch
    chunks = list(summarize_text(long_text, "test-hf-key"))

    assert chunks == ["An error occurred while generating the summary."]
```

- [ ] **Step 2: Run to verify it already passes**

Run: `pytest tests/test_summarization.py -v -k chunked`
Expected: PASS (Task 6's implementation already wraps both branches identically)

If it fails, the chunked branch's `try/except` in `summarization.py` needs the same `return` after yielding the error message as the single-request branch — add it.

- [ ] **Step 3: Commit**

```bash
git add tests/test_summarization.py
git commit -m "test: cover chunked summarization failure path"
```

---

### Task 11: audio_input.py — browser capture, idempotency, malformed-blob handling

**Files:**
- Modify: `src/audio/audio_input.py`
- Test: `tests/test_audio_input.py`

**Interfaces:**
- Produces:
  - `get_audio_input(audio_bytes: bytes) -> str | None`
  - `has_processed(session_state: dict, audio_bytes: bytes) -> bool`
  - `mark_processed(session_state: dict, audio_bytes: bytes) -> None`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_audio_input.py
import io


def test_has_processed_false_before_marking():
    from audio.audio_input import has_processed

    session_state = {}
    assert has_processed(session_state, b"some-wav-bytes") is False


def test_mark_processed_then_has_processed_true():
    from audio.audio_input import has_processed, mark_processed

    session_state = {}
    mark_processed(session_state, b"some-wav-bytes")
    assert has_processed(session_state, b"some-wav-bytes") is True


def test_has_processed_false_for_different_bytes():
    from audio.audio_input import has_processed, mark_processed

    session_state = {}
    mark_processed(session_state, b"first-recording")
    assert has_processed(session_state, b"second-recording") is False


def test_get_audio_input_returns_none_on_malformed_bytes():
    from audio.audio_input import get_audio_input

    assert get_audio_input(b"not a real wav file") is None


def test_get_audio_input_returns_none_on_empty_bytes():
    from audio.audio_input import get_audio_input

    assert get_audio_input(b"") is None


def test_get_audio_input_returns_recognized_text(monkeypatch):
    from audio import audio_input as audio_input_module

    class FakeRecognizer:
        def record(self, source):
            return "fake-audio-data"

        def recognize_google(self, audio):
            return "hello world"

    class FakeAudioFile:
        def __init__(self, source):
            pass

        def __enter__(self):
            return "fake-source"

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(audio_input_module.sr, "Recognizer", FakeRecognizer)
    monkeypatch.setattr(audio_input_module.sr, "AudioFile", FakeAudioFile)

    result = audio_input_module.get_audio_input(b"RIFF....WAVEfmt ")

    assert result == "hello world"
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_audio_input.py -v`
Expected: FAIL (`get_audio_input` still takes `duration` and uses `sr.Microphone`; `has_processed`/`mark_processed` don't exist)

- [ ] **Step 3: Implement**

```python
# src/audio/audio_input.py
import hashlib
import io

import speech_recognition as sr

PROCESSED_HASH_KEY = "audio_input_last_processed_hash"


def get_audio_input(audio_bytes: bytes) -> str | None:
    """Recognizes speech from an in-memory WAV blob (browser-captured, no PyAudio)."""
    if not audio_bytes:
        return None

    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio = recognizer.record(source)
    except Exception:
        return None

    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return None
    except sr.RequestError:
        return None


def has_processed(session_state: dict, audio_bytes: bytes) -> bool:
    """Guards against re-processing the same sticky st.audio_input value on an unrelated rerun."""
    current_hash = hashlib.sha256(audio_bytes).hexdigest()
    return session_state.get(PROCESSED_HASH_KEY) == current_hash


def mark_processed(session_state: dict, audio_bytes: bytes) -> None:
    session_state[PROCESSED_HASH_KEY] = hashlib.sha256(audio_bytes).hexdigest()
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_audio_input.py -v`
Expected: PASS

- [ ] **Step 5: Update the package init**

`src/audio/__init__.py` already exports `get_audio_input`; no signature-name change needed, only its call sites in `app.py` (Task 13).

- [ ] **Step 6: Commit**

```bash
git add src/audio/audio_input.py tests/test_audio_input.py
git commit -m "feat: rebuild audio input around browser capture, drop PyAudio"
```

---

### Task 12: audio_output.py — return bytes instead of playing them

**Files:**
- Modify: `src/audio/audio_output.py`
- Test: `tests/test_audio_output.py`

**Interfaces:**
- Produces: `speak_text(text: str) -> bytes` — MP3 bytes, or `b""` on failure/empty input.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_audio_output.py
def test_speak_text_returns_bytes(monkeypatch):
    from audio import audio_output as audio_output_module

    class FakeTTS:
        def __init__(self, text, lang):
            self.text = text

        def write_to_fp(self, fp):
            fp.write(b"fake-mp3-bytes")

    monkeypatch.setattr(audio_output_module, "gTTS", FakeTTS)

    result = audio_output_module.speak_text("hello world")

    assert result == b"fake-mp3-bytes"


def test_speak_text_returns_empty_bytes_for_empty_input():
    from audio.audio_output import speak_text

    assert speak_text("") == b""


def test_speak_text_returns_empty_bytes_on_tts_failure(monkeypatch):
    from audio import audio_output as audio_output_module

    class FailingTTS:
        def __init__(self, text, lang):
            raise ValueError("tts service unavailable")

    monkeypatch.setattr(audio_output_module, "gTTS", FailingTTS)

    result = audio_output_module.speak_text("hello world")

    assert result == b""
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_audio_output.py -v`
Expected: FAIL (`speak_text` still uses `pygame` and returns `None`)

- [ ] **Step 3: Implement**

```python
# src/audio/audio_output.py
from gtts import gTTS
from io import BytesIO


def speak_text(text: str) -> bytes:
    """Synthesizes speech and returns MP3 bytes for browser playback via st.audio."""
    if not text:
        return b""

    try:
        tts = gTTS(text=text, lang="en")
        buffer = BytesIO()
        tts.write_to_fp(buffer)
        return buffer.getvalue()
    except Exception:
        return b""
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_audio_output.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/audio/audio_output.py tests/test_audio_output.py
git commit -m "feat: audio output returns bytes for browser playback, drop pygame"
```

---

### Task 13: app.py — wire everything together

**Files:**
- Modify: `src/app.py` (full task-branch rewrite)

**Interfaces:**
- Consumes: `credentials.get_gemini_key`, `credentials.get_huggingface_key`, `credentials.missing_key_message`; `nlp.RAG.init_RAG`, `qa`, `create_vector_store`, `create_qa_model`; `nlp.vector_cache.compute_files_fingerprint`, `get_cached_vector_store`, `store_vector_store`; `nlp.summarization.summarize_text`; `cv.image_captioning.caption_image`; `audio.audio_input.get_audio_input`, `has_processed`, `mark_processed`; `audio.audio_output.speak_text`.

This is the integration task — no new pure logic, so no new unit tests; verified manually per the checklist in Step 6.

- [ ] **Step 1: Gate RAG initialization on the Gemini key**

Replace the unconditional module-load-time `init_RAG()` call:

```python
# before
llm, embedding_model, prompt_template, contextualize_q_prompt = init_RAG()
```

```python
# after — moved out of module scope, called only where needed with the live key
from nlp import init_RAG, create_vector_store, create_qa_model, qa
from nlp.vector_cache import compute_files_fingerprint, get_cached_vector_store, store_vector_store
```

(No RAG objects are built until a Gemini key is present and the QA task branch runs — see Step 4.)

- [ ] **Step 2: Image Captioning branch — gate on HuggingFace key**

```python
elif task_name == "Image Captioning":
    hf_key = get_huggingface_key(st.session_state)
    uploaded_files = st.file_uploader(
        "Upload a file", type=["jpg", "jpeg", "png"], accept_multiple_files=False
    )
    st.divider()

    if not hf_key:
        st.info(missing_key_message("Image Captioning", "HuggingFace"))
    elif uploaded_files:
        col1, col2 = st.columns(2)
        _, col21, col22 = col2.columns([1, 6, 6])

        col1.image(uploaded_files, caption="Uploaded Image", use_container_width=True)

        if col21.button("Caption Image"):
            st.session_state.image_caption = caption_image(uploaded_files, hf_key)
        if st.session_state.image_caption:
            col2.write(f"**Caption:** {st.session_state.image_caption}")

        if col22.button("Audio Output", key="audio_image_caption"):
            audio_bytes = speak_text(st.session_state.image_caption)
            if audio_bytes:
                col2.audio(audio_bytes, format="audio/mp3", autoplay=True)
    else:
        st.write("Please upload an image file for captioning.")
```

- [ ] **Step 3: Text Summarization branch — gate on HuggingFace key, real streaming, browser audio**

```python
elif task_name == "Text Summarization":
    hf_key = get_huggingface_key(st.session_state)
    if not hf_key:
        st.info(missing_key_message("Text Summarization", "HuggingFace"))
    else:
        text_input = st.radio("Select the input type:", ["Text", "File", "Audio"])

        if text_input == "Audio":
            audio_blob = st.audio_input("Record text to summarize")
            if audio_blob and not has_processed(st.session_state, audio_blob.getvalue()):
                mark_processed(st.session_state, audio_blob.getvalue())
                recognized = get_audio_input(audio_blob.getvalue())
                if recognized:
                    st.session_state.audio_input = recognized.capitalize().strip() + "."
                    st.write(f"**You (audio):** {st.session_state.audio_input}")
                    st.session_state.text_summarization = st.write_stream(
                        summarize_text(recognized, hf_key)
                    )
                else:
                    st.error("Couldn't understand that recording — please try again.")

        elif text_input == "Text":
            if query := st.text_area("Enter a text for summarization:"):
                st.session_state.text_summarization = st.write_stream(summarize_text(query, hf_key))
            else:
                st.write_stream(custom_message_generator("Please enter a text for summarization."))

        else:
            uploaded_files = st.file_uploader(
                "Upload a file", type=["pdf", "csv", "txt", "md"], accept_multiple_files=False
            )
            if uploaded_files:
                text = read_file(uploaded_files)
                if text == "Unsupported file type.":
                    st.write_stream(custom_message_generator(text))
                else:
                    st.session_state.text_summarization = st.write_stream(summarize_text(text, hf_key))
            else:
                st.write_stream(custom_message_generator("Please upload a file for summarization."))

        if st.button("Audio Output", key="audio_text_summarization"):
            summary_text = (
                st.session_state.text_summarization.split("**Summary:**")[-1].strip()
                if st.session_state.text_summarization
                else "No response to output."
            )
            audio_bytes = speak_text(summary_text)
            if audio_bytes:
                st.audio(audio_bytes, format="audio/mp3", autoplay=True)
```

- [ ] **Step 4: Question Answering branch — build RAG lazily, cached vector store, real streaming, browser audio**

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
            fingerprint = compute_files_fingerprint(uploaded_files)
            vector_store = get_cached_vector_store(st.session_state, fingerprint)
            if vector_store is None:
                with st.spinner("Indexing files…"):
                    vector_store = create_vector_store(uploaded_files, embedding_model)
                    store_vector_store(st.session_state, fingerprint, vector_store)
            else:
                st.caption("Using cached index for this file set.")

            qa_model = create_qa_model(vector_store, llm, prompt_template, contextualize_q_prompt)

        for message in st.session_state.messages:
            avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        if query := st.chat_input():
            st.chat_message("user", avatar="🧑‍💻").markdown(query)
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Wait for the response..."):
                    if qa_model:
                        response = st.write_stream(qa(query, qa_model, st.session_state.messages))
                    else:
                        response = st.write_stream(
                            custom_message_generator("Please upload a file to start the chat.")
                        )

            if qa_model and response:
                st.session_state.messages.append({"role": "user", "content": query})
                st.session_state.messages.append({"role": "assistant", "content": response})

            if random.random() > 0.9 and response and qa_model:
                st.balloons()

        _, col1, col2, col3 = st.columns([1, 3, 3, 3])

        if col1.button("Start New Chat"):
            st.session_state.messages = []
            st.rerun()

        with col2:
            audio_blob = st.audio_input("Ask by voice")
        if audio_blob and not has_processed(st.session_state, audio_blob.getvalue()):
            mark_processed(st.session_state, audio_blob.getvalue())
            recognized = get_audio_input(audio_blob.getvalue())
            if recognized:
                recognized = recognized.capitalize()
                st.chat_message("user", avatar="🧑‍💻").markdown(recognized)
                with st.chat_message("assistant", avatar="🤖"):
                    with st.spinner("Wait for the response..."):
                        if qa_model:
                            response = st.write_stream(qa(recognized, qa_model, st.session_state.messages))
                        else:
                            response = st.write_stream(
                                custom_message_generator("Please upload a file to start the chat.")
                            )
                if qa_model and response:
                    st.session_state.messages.append({"role": "user", "content": recognized})
                    st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                st.error("Couldn't understand that recording — please try again.")

        if col3.button("Audio Output", key="audio_qa"):
            try:
                last_message = st.session_state.messages[-1]
            except IndexError:
                last_message = {"role": "assistant", "content": "No response to output."}

            audio_response = (
                last_message["content"] if last_message["role"] == "assistant" else "No response to output."
            )
            audio_bytes = speak_text(audio_response)
            if audio_bytes:
                col3.audio(audio_bytes, format="audio/mp3", autoplay=True)
```

- [ ] **Step 5: Update imports at the top of app.py**

```python
from audio import get_audio_input, speak_text, has_processed, mark_processed
from cv import caption_image
from nlp import summarize_text, init_RAG, create_vector_store, create_qa_model, qa
from nlp.vector_cache import compute_files_fingerprint, get_cached_vector_store, store_vector_store
from utils import read_file, custom_message_generator
from credentials import render_key_sidebar, get_gemini_key, get_huggingface_key, missing_key_message
from paths import asset_path
```

Update `src/audio/__init__.py` and `src/nlp/__init__.py` to export the new names:

```python
# src/audio/__init__.py
from .audio_input import get_audio_input, has_processed, mark_processed
from .audio_output import speak_text
```

```python
# src/nlp/__init__.py
from .RAG import init_RAG, create_vector_store, create_qa_model, qa, build_chat_history
from .summarization import summarize_text
```

- [ ] **Step 6: Manual verification checklist (requires real Gemini + HuggingFace keys)**

Run: `streamlit run src/app.py`

1. With no keys entered: every task shows its "enter a key" prompt; nothing crashes.
2. Enter a Gemini key: Question Answering accepts a file upload, indexes it (spinner shown once), answers stream in incrementally (not all-at-once), re-uploading the same file shows "Using cached index" instead of re-indexing.
3. Click "Ask by voice", record a short question, confirm it's transcribed and answered exactly once — then click "Start New Chat" and confirm the same recording is *not* reprocessed.
4. Click "Audio Output" and confirm playback starts in the browser.
5. Enter a HuggingFace key: Image Captioning and Text Summarization (Text/File/Audio inputs) all produce real output; Audio Output plays back correctly for each.
6. Grep for leftover dead patterns:

```bash
grep -n "itertools.tee\|time.sleep(0.5)\|pygame\|pyaudio\|st.secrets\[" src/app.py src/nlp/*.py src/audio/*.py src/cv/*.py
```

Expected: no matches.

- [ ] **Step 7: Commit**

```bash
git add src/app.py src/audio/__init__.py src/nlp/__init__.py
git commit -m "feat: wire BYOK gating, real streaming, and browser audio into app.py"
```

---

### Task 14: Dependency slimming

**Files:**
- Modify: `requirements.txt` (regenerated)
- Modify: `packages.txt`
- Modify: `.devcontainer/devcontainer.json`

- [ ] **Step 1: Regenerate requirements.txt from a clean venv**

```bash
python3 -m venv /tmp/nexchat-clean-venv
source /tmp/nexchat-clean-venv/bin/activate
pip install --upgrade pip
pip install streamlit langchain langchain-community langchain-google-genai \
            langchain-core faiss-cpu PyPDF2 pandas requests python-dotenv \
            SpeechRecognition gTTS
pip freeze > /home/momad/Projects/NexChat/requirements.txt
deactivate
rm -rf /tmp/nexchat-clean-venv
```

- [ ] **Step 2: Verify torch and audio-device packages are gone**

```bash
grep -iE "^torch|^sentence-transformers|^pyaudio|^pygame|^chime|^langchain-huggingface|^transformers" requirements.txt
```

Expected: no output.

- [ ] **Step 3: Empty packages.txt**

```bash
: > packages.txt
```

`portaudio19-dev`, `python3-pyaudio`, `pulseaudio-utils` were only needed for server-side mic capture, now removed. `ffmpeg`, `libsndfile1`, `libasound2-dev` were also audio-device support; confirm nothing else in the codebase shells out to `ffmpeg` before deleting those lines too:

```bash
grep -rn "ffmpeg\|libsndfile\|libasound" src/
```

Expected: no matches — safe to leave `packages.txt` empty.

- [ ] **Step 4: Trim devcontainer.json**

In `.devcontainer/devcontainer.json`, change:

```json
"updateContentCommand": "sudo apt update && sudo apt upgrade -y && sudo apt install -y portaudio19-dev python3-pyaudio ffmpeg libsndfile1 && [ -f requirements.txt ] && pip3 install --user -r requirements.txt && echo '✅ Audio dependencies and requirements installed'",
```

to:

```json
"updateContentCommand": "sudo apt update && sudo apt upgrade -y && [ -f requirements.txt ] && pip3 install --user -r requirements.txt && echo '✅ Requirements installed'",
```

- [ ] **Step 5: Verify a fresh install still runs the test suite**

```bash
python3 -m venv /tmp/nexchat-verify-venv
source /tmp/nexchat-verify-venv/bin/activate
pip install -r requirements-dev.txt
pytest tests/ -v
deactivate
rm -rf /tmp/nexchat-verify-venv
```

Expected: all tests pass without torch ever being installed.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt packages.txt .devcontainer/devcontainer.json
git commit -m "chore: drop torch/PyAudio/pygame from deploy footprint"
```

---

### Task 15: Deployment smoke test (manual, Streamlit Community Cloud)

**Files:** none — verification only.

- [ ] **Step 1: Push the branch and open a Community Cloud app pointed at it**

Follow Streamlit Community Cloud's "New app" flow against this repo/branch.

- [ ] **Step 2: Confirm the build succeeds under the resource ceiling**

Expected: build completes without an out-of-memory/disk error (the prior torch-based build was the likely cause of any prior failure — this is now removed).

- [ ] **Step 3: Confirm cold boot with zero keys**

Expected: app loads to the sidebar with both key fields empty and every task showing its "enter a key" prompt — no crash, matching the local manual check from Task 13.

- [ ] **Step 4: Re-run the Task 13 Step 6 manual checklist against the deployed URL**

Same six checks — RAG upload/cache/stream, voice input idempotency, voice output playback, captioning, summarization (all three input modes) — this time from a browser hitting the Cloud deployment rather than localhost, to confirm the audio browser-widget behavior (which depends on browser mic/speaker permissions, not local process audio) actually works end-to-end.

- [ ] **Step 5: Report results back**

No commit — this task's output is a pass/fail report, not a code change. If any step fails, file it against the relevant task above rather than patching ad hoc.

---

## Self-Review Notes

- **Spec coverage:** every numbered spec section (§4.1–§4.10) maps to a task above; the parity matrix (§5) is exercised by Task 13/15's manual checklists; the testing plan (§6) items — missing-key boot, key invalidation mid-conversation, audio idempotency, autoplay — are covered by Task 13 Step 6 (idempotency, boot) and Task 15 Step 4 (autoplay, full deployed parity). Mid-conversation key invalidation (§4.10) has no separate task: it degrades to the mid-stream error path already unit tested in Task 9 (`test_qa_yields_interruption_message_mid_stream` is exactly "the key stops working partway through a stream"), so no new mechanism is needed.
- **Placeholder scan:** no TBD/TODO; every step has real code, not a description of code.
- **Type consistency:** `qa_model.stream()` chunk shape (`{"answer": ...}`) matches LangChain's `create_retrieval_chain` output contract used consistently in Tasks 5, 8, 9, 13. `summarize_text`/`caption_image` signatures (`text, huggingface_api_key` / `uploaded_file, huggingface_api_key`) match between Task 6's implementation and Task 13's call sites. `get_audio_input(audio_bytes)`, `has_processed`/`mark_processed(session_state, audio_bytes)` match between Task 11 and Task 13. `speak_text(text) -> bytes` matches between Task 12 and Task 13's `st.audio(audio_bytes, ...)` calls.
