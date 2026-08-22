# NexChat: Streamlit Community Cloud + BYOK Design

**Status:** Draft for review
**Date:** 2026-08-22

## 1. Goal

Take NexChat from "worked once on one Windows machine with the author's
own keys baked in" to a codebase that:

1. Runs identically on Streamlit Community Cloud and on any local OS
   (Windows/macOS/Linux), with no feature silently dropped between the
   two.
2. Uses **BYOK** (bring-your-own-key): every visitor supplies their own
   Gemini and HuggingFace API keys via the sidebar; the app holds no
   credentials of its own and boots successfully with zero keys
   present.
3. Fits Community Cloud's ~1GB resource ceiling.
4. Fixes the correctness bugs and dead dependencies found during the
   audit, without adding scope beyond what "fully working" requires.

## 2. Problem Inventory

Findings from reading `app.py`, `utils.py`, `RAG.py`,
`summarization.py`, `image_captioning.py`, `audio_input.py`,
`audio_output.py`, `requirements.txt`, `packages.txt`, and
`.devcontainer/devcontainer.json`, plus two endpoint probes.

| # | Finding | File:Line | Severity |
|---|---|---|---|
| 1 | Server-side mic capture (PyAudio) and server-side playback (pygame) — no mic/speaker exists on Cloud | `audio_input.py`, `audio_output.py` | Blocker |
| 2 | Credentials read at **import time** via `dotenv_values(find_dotenv())` with `st.secrets` fallback — crashes on import with no keys, incompatible with per-user BYOK | `RAG.py:22-29`, `summarization.py:11-14`, `image_captioning.py:10-13` | Blocker |
| 3 | `HuggingFaceEmbeddings("BAAI/bge-small-en-v1.5")` runs a local sentence-transformers model, pulling in `torch` (~800MB) though `torch`/`transformers` are never imported directly anywhere in `src/` | `RAG.py:64` | Blocker (deploy size) |
| 4 | Legacy endpoint `api-inference.huggingface.co` returns **no response at all** (`curl` → connection failure); the current endpoint is `router.huggingface.co/hf-inference/models/...` (confirmed alive, returns 401 without a key) | `summarization.py:7`, `image_captioning.py:7` | Blocker (feature is dead today, independent of deployment) |
| 5 | Vector store rebuilt from scratch on **every Streamlit rerun** (every chat message), with no caching — burns the user's own embedding quota under BYOK | `app.py:172` | High |
| 6 | `itertools.tee` + fully draining one branch before `st.write_stream` on the other — the whole response is generated, *then* replayed as a fake typewriter; QA additionally sleeps 0.1s/word and summarization 0.5s/sentence, adding pure latency | `app.py:106-108,128-130,151-153,200-202,245-247`; `RAG.py:246-250`; `summarization.py:63` | High (UX) |
| 7 | `get_audio_input(duration)` ignores its `duration` argument and always records the module constant `DURATION=5`, so the 30s summarization recording actually records 5s | `audio_input.py:6,20` | High (moot once audio is rebuilt, listed for completeness) |
| 8 | `qa()`'s chat-history loop reassigns `user_message`/`ai_answer` each iteration instead of accumulating — only the single most recent turn ever reaches the model, and if the most recent message is from the user (no `ai_answer` bound yet), the resulting `NameError` is silently swallowed by `except Exception: pass` | `RAG.py:229-238` | Medium (correctness) |
| 9 | Image/logo paths are CWD-relative (`"./imgs/icon.png"`); only resolve when the process's CWD happens to be the repo root | `app.py:37,47` | Medium (cross-platform) |
| 10 | Sidebar links point at the old GitLab repo and a defunct co-author link structure; project is now on GitHub | `app.py:56-67` | Low |
| 11 | `.devcontainer/devcontainer.json` installs `portaudio19-dev`/`python3-pyaudio`/`pulseaudio-utils` and disables CORS/XSRF protection; both become unnecessary/undesirable once audio moves to the browser | `.devcontainer/devcontainer.json` | Low |
| 12 | Several `except` clauses silently swallow errors with no user-facing message — true bare `except:` at `audio_output.py:30`, `summarization.py:42,50`; `except Exception as e: pass` at `RAG.py:237,243` | multiple | Medium (violates comprehensive-error-handling convention) |

Not a bug, confirmed safe: `from utils import read_file`-style imports
resolve via `sys.path[0]` (Python adds the launched script's directory
automatically), so they work regardless of CWD — only the *image
paths* above are the real CWD hazard.

## 3. Key Decisions (already made with you)

- **Audio**: rebuilt entirely in-browser. Input via `st.audio_input`
  (browser mic → WAV bytes back to the server); output via `st.audio`
  (server generates MP3 bytes, browser plays them). No server-side
  device access at all, so behavior is identical locally and deployed.
- **Embeddings**: move off local `HuggingFaceEmbeddings` to a **hosted**
  embedding call, eliminating `torch`/`sentence-transformers`
  entirely. Since Gemini is already one of the two BYOK keys, use
  `GoogleGenerativeAIEmbeddings` (`models/text-embedding-004`) — no
  new key type introduced.
- **BYOK scope**: two keys — **Gemini** (chat/RAG + embeddings) and
  **HuggingFace** (summarization + captioning). Speech-to-text stays
  on the keyless `speech_recognition` Google Web Speech endpoint,
  unchanged in provider, but re-targeted from a live microphone
  (`sr.Microphone`) to an in-memory WAV (`sr.AudioFile`) — which also
  means **PyAudio is no longer needed at all** (it's only required for
  live device capture, not for recognizing an existing audio buffer).
- **Key persistence**: session-only (`st.session_state`), never
  written to disk, never logged. Locally, an optional `.env` is read
  once at startup purely to *prefill* the sidebar fields so you don't
  retype keys during development; the deployed app has no `.env` and
  the fields start empty.

## 4. Architecture

### 4.1 Credential layer (replaces all three import-time reads)

New module `src/credentials.py`:

- `get_gemini_key() -> str | None` and `get_huggingface_key() -> str |
  None` read from `st.session_state["gemini_api_key"]` /
  `["huggingface_api_key"]`.
- `init_credential_state()` runs once per session: seeds
  `st.session_state` from a local `.env` if one exists (via
  `python-dotenv`, guarded so a missing file is not an error), else
  leaves the fields empty.
- Sidebar renders two `st.text_input(..., type="password")` fields
  bound to those session-state keys, plus a status line ("✅ Gemini
  connected" / "⚠️ Enter a HuggingFace key to use Summarization and
  Captioning").

Every provider call site changes from *module-level client
construction* to a **lazy factory function** called at the point of
use, e.g. `init_llm_model(api_key: str)` instead of reading
`os.environ` at import. `RAG.py`, `summarization.py`,
`image_captioning.py` all lose their top-level `try/except
dotenv/st.secrets` blocks. Missing key → the calling UI code shows a
clear inline message ("Enter your Gemini API key in the sidebar to use
this feature") instead of the feature crashing the whole app.

**Hard rule: never mutate `os.environ` with a per-request key.**
Community Cloud can serve multiple concurrent sessions from a single
shared worker process; `os.environ["GOOGLE_API_KEY"] = ...` (the
current pattern in `RAG.py:26,29`) is *process-global* state, so one
user's key can leak into another user's in-flight request. Every
client — `ChatGoogleGenerativeAI`, `GoogleGenerativeAIEmbeddings`, and
the HuggingFace `requests` calls — must take the key as a constructor
argument or request parameter (`google_api_key=...`,
`Authorization` header built per-call), never via environment
mutation. This applies equally to `summarization.py`'s and
`image_captioning.py`'s module-level `headers = {"Authorization": f"Bearer
{huggingface_api_key}"}` dict — that also becomes a per-call local,
built inside the function from the caller-supplied key, not a
module-level value computed once at import.

### 4.2 Embeddings & vector store

- `init_embeddings_model(gemini_key)` returns
  `GoogleGenerativeAIEmbeddings(model="models/text-embedding-004",
  google_api_key=gemini_key)`.
- **Caching must be session-scoped, not `st.cache_resource`.**
  `st.cache_resource` is process-global in Streamlit — Community Cloud
  can serve multiple users from one shared process, and the cached
  object here is a `FAISS` store holding a closure over the specific
  `embedding_model` instance (and therefore the specific Gemini key)
  used to build it. Keying purely by file fingerprint + model id (no
  key material) means a second user who happens to upload a
  same-name/same-size/same-hash file — trivial with any shared public
  document — would receive the first user's cached store and start
  running their retrieval queries against the first user's embedding
  model, i.e. against the first user's key and quota, without either
  user knowing. That is a direct violation of the "session-only,
  never leaves this session" key promise (§3), so `cache_resource` is
  not used here at all. Instead: `create_vector_store` result is
  stored directly in `st.session_state["vector_store"]` alongside the
  fingerprint (file names + sizes + sha256) it was built from. On
  rerun, if the current upload's fingerprint matches the stored one,
  reuse the stored store; otherwise rebuild and overwrite. This is
  naturally bounded to a single entry per session (no growing
  collection, no `ttl`/`max_entries` tuning needed) and never crosses
  session boundaries, eliminating the leak by construction rather than
  by hashing key material into a cache key.

### 4.3 HuggingFace endpoint migration

Both `summarization.py` and `image_captioning.py` move their
`API_URL` from `api-inference.huggingface.co/models/...` to
`router.huggingface.co/hf-inference/models/...` (verified live via
`curl`, currently returning `401` — correct "needs auth" behavior).

That `401` only confirms the router itself is reachable — it does
**not** confirm that `facebook/bart-large-cnn` and
`Salesforce/blip-image-captioning-base` are specifically still served
under the `hf-inference` provider path, since HF's auth check can fire
before any model-existence check. HF has been consolidating its free
serverless inference behind an "Inference Providers" marketplace where
not every legacy model remains available under every provider.
**Verifying both specific models return a real response with a valid
key is a pre-implementation task**, done before other work depends on
it, not deferred as generic "testing." Contingency if a model is gone:
substitute the closest currently-served equivalent on the same
provider (e.g. another BART/T5 summarization model,
another BLIP/vit-gpt2 captioning model) rather than treating the
feature as blocked — the endpoint migration's intent (get these two
features working again) doesn't depend on the exact model names, only
on *a* working model of the same kind.

### 4.4 Audio, rebuilt

- **Input**: `st.audio_input("Record your question")` returns an
  `UploadedFile`-like WAV blob. `get_audio_input(audio_bytes: bytes) ->
  str | None` wraps it in `sr.AudioFile`, feeds it to
  `recognizer.recognize_google`, and returns text or `None` on
  failure — same recognizer, new source. `chime` (audible cue on a
  *server* speaker) is removed; a `st.toast`/`st.spinner` gives the
  equivalent feedback in-browser.
  - **Rerun idempotency is required, not optional.** `st.audio_input`'s
    return value is sticky across reruns once populated — unlike the
    current button-triggered flow, there is no natural "only fires on
    click" gate. Without a guard, any unrelated widget interaction
    (e.g. clicking "Start New Chat") triggers a rerun that sees the
    same still-populated audio value and silently reprocesses it —
    re-running recognition and re-invoking `qa()`/`summarize_text()`
    against stale input, burning the user's key quota again for
    nothing. Fix: track the last-processed blob's identity (e.g. sha256
    of its bytes) in `st.session_state`; skip processing if the
    current value's identity matches.
  - **Malformed/empty blobs must be handled at the file-open layer,
    not just the recognizer layer.** The existing exception handling
    (`sr.UnknownValueError`, `sr.RequestError`) only covers
    recognition failures. A zero-length or truncated WAV — e.g. a user
    tapping record then immediately stop — can raise from *opening*
    the file inside `sr.AudioFile` (stdlib `wave` errors) before
    recognition is ever reached; that path needs its own catch with a
    user-facing "couldn't read that recording, try again" message.
- **Output**: `speak_text(text: str) -> bytes` returns the gTTS MP3
  bytes directly instead of playing them; call sites do
  `st.audio(speak_text(text), format="audio/mp3", autoplay=True)`.
  `pygame` is removed.
- `src/audio/audio_input.py` and `audio_output.py` keep their names
  and roles (capture text from audio, render audio from text) — only
  the transport (device vs. browser) changes, so the rest of `app.py`
  barely changes shape.

### 4.5 Real streaming

- **QA**: `qa_model` is a LangChain `Runnable`; swap
  `qa_model.invoke(...)` for `qa_model.stream(...)` and yield
  `chunk["answer"]` pieces as they arrive from Gemini — genuine
  token-level streaming, no `time.sleep`.
- **Summarization**: the HuggingFace inference call is a single
  request/response (not a token stream), so there is nothing to stream
  from the API. The fix here is narrower: stop pre-draining the
  generator through `itertools.tee` before display — yield sentences
  directly into `st.write_stream` as they're produced by the local
  split, with the artificial `time.sleep(0.5)` removed (the wait was
  purely cosmetic pacing, not required for correctness).
- Net effect: responses start appearing as soon as data exists, not
  after the full response has already been computed once.

**Control-flow change this requires.** The current code decides which
generator to hand `st.write_stream` *before* streaming starts (`if
response: ... else: generator = custom_message_generator(error)`,
then `st.write_stream(generator)`) — but genuine streaming means the
outcome isn't known until the stream is drained, so that up-front
branch no longer works. Both the QA and summarization call sites
change to the idiomatic Streamlit pattern: `response =
st.write_stream(generator)` (which streams to the UI *and* returns the
fully concatenated text), then branch on `response` afterward to
decide whether to append it to `st.session_state.messages` or show an
error. This is a real rewrite of both call sites' control flow, not a
one-line `invoke()` → `stream()` swap.

**Mid-stream errors need their own handling.** Today a single
`try/except` wraps one blocking `qa_model.invoke()` call. Under
`.stream()`, a key that's revoked mid-generation, or a transient
network error, raises from *inside* the iteration — after some chunks
may already have reached `st.write_stream`. A `try/except` around the
whole call can no longer cleanly produce "show one error message";
instead, the generator itself catches exceptions raised during
iteration and yields a final "⚠️ Response interrupted: could not
reach Gemini" chunk rather than letting the exception propagate as a
raw traceback into the UI. This applies to both the QA and
summarization streaming paths, and is planned together with §4.7's
error-handling cleanup rather than as a separate concern — it's new
failure surface area §4.7 doesn't cover on its own.

### 4.6 Paths

`imgs/icon.png` and `imgs/logo.png` resolve via
`Path(__file__).resolve().parent.parent / "imgs" / "..."` (or an
equivalent constants module) instead of a CWD-relative string, so both
`streamlit run src/app.py` (from repo root) and `cd src && streamlit
run app.py` (the devcontainer's convention) work identically.

### 4.7 Bug fixes folded in

- `qa()`'s chat-history construction rewritten to accumulate proper
  `(HumanMessage, AIMessage)` pairs across the full loop instead of
  only keeping the last iteration's variables.
- Bare `except:`/`except Exception: pass` blocks replaced with
  targeted exception handling that surfaces a user-facing message
  (e.g. `st.error(...)`) and, where useful, logs the underlying
  exception — per the no-silent-swallow convention.
- This is planned together with §4.5's mid-stream error handling, not
  independently: the streaming refactor introduces new places
  exceptions can surface (mid-iteration rather than around one
  blocking call), and both need one consistent "user sees a clear
  inline message, nothing crashes, nothing is silently dropped"
  design rather than two uncoordinated passes.

### 4.8 Dependency slimming

Removed entirely: `torch`, `sentence-transformers` (transitive),
`PyAudio`, `pygame`, `chime`. Kept: `SpeechRecognition` (file-mode
only — no PyAudio dependency needed for `sr.AudioFile`), `gTTS`,
`langchain-google-genai`, `FAISS`, `PyPDF2`, `pandas`, `requests`.
`packages.txt` (apt packages) becomes empty or is deleted;
`.devcontainer/devcontainer.json`'s `updateContentCommand` drops the
`portaudio19-dev`/`python3-pyaudio`/`pulseaudio-utils` installs.

`requirements.txt` is regenerated from a clean virtualenv (`pip
install` the trimmed direct dependencies, then `pip freeze`) rather
than hand-deleting lines — `torch`/`sentence-transformers` also drag
in `transformers`, `tokenizers`, `safetensors`, `huggingface-hub`
(pinned to a torch-compatible version), `scikit-learn`, `scipy`,
`sympy`, `mpmath`, and `networkx`, and manual deletion risks leaving
several of these stranded unnoticed.

### 4.9 UI/UX improvements

- Sidebar: masked key inputs with a live connected/missing status per
  provider, replacing the static author/repo links block (which moves
  to a collapsed `st.expander("About")` at the bottom so it doesn't
  compete with the keys for attention).
- Each task view checks its required key(s) up front and shows a
  single clear prompt ("Enter your HuggingFace key to caption images")
  instead of the feature silently failing partway through.
- File upload feedback: show the cached/recomputing state of the
  vector store ("Using cached index for 2 files" vs. "Indexing 2
  files…") so BYOK users understand when they're spending quota.
- Update the GitHub repo link; drop the stale GitLab reference.
- Keep the existing task-selection / layout structure (`selectbox` +
  branches, columns, chat UI) — it works and a UI framework rewrite is
  out of scope; polish within it rather than restructure it.

### 4.10 Mid-conversation key invalidation

If a key that was valid becomes invalid partway through a session
(revoked, expired, quota exhausted), the failure surfaces through the
mid-stream error handling in §4.5/§4.7 as an inline error on that one
exchange — it does **not** clear or roll back
`st.session_state.messages`. Prior successful exchanges stay visible;
only the failed turn shows an error state, and the user can update
their key in the sidebar and retry without losing chat history.

## 5. Per-Feature Parity Matrix

| Feature | Local (any OS) | Streamlit Community Cloud | Notes |
|---|---|---|---|
| Text Q&A / RAG chat | ✅ | ✅ | Identical code path; embeddings hosted, so no local-only model download |
| File upload (PDF/CSV/TXT/MD) for RAG | ✅ | ✅ | Unchanged; already pure Python parsing |
| Image captioning | ✅ | ✅ | Requires HF key; re-pointed to live endpoint |
| Text summarization (text / file input) | ✅ | ✅ | Requires HF key; re-pointed to live endpoint |
| Voice input (question or summarization source) | ✅ | ✅ | Browser mic via `st.audio_input`; identical on both |
| Voice output (spoken answers/captions/summaries) | ✅ | ✅ | Browser playback via `st.audio`; identical on both |
| BYOK key entry | ✅ | ✅ | Session-state sidebar fields on both |
| Local key convenience (`.env` autofill) | ✅ | N/A (no `.env` present) | Dev-only convenience, not a feature gap |

No feature present in the current app is dropped; the two audio
features change *transport* (device → browser) but remain fully
functional in both environments, which is the property that was
previously impossible.

## 6. Testing Plan

- **Unit-level**: `read_file`/`read_csv`/`read_pdf` parsing (existing
  behavior, unchanged — regression only), `create_sentences`,
  vector-store fingerprint/caching logic, chat-history pair
  construction in `qa()`.
- **Integration, with real keys (manual, since these are paid/keyed
  external APIs)**: one live call per provider path — Gemini chat,
  Gemini embeddings, HF summarization, HF captioning, `st.audio_input`
  → `sr.AudioFile` recognition, `gTTS` → `st.audio` playback — run
  once locally before considering the migration done.
- **Missing-key behavior**: app boots with zero keys set and every
  feature shows its "enter a key" prompt rather than crashing —
  this is the core BYOK correctness property and gets explicit
  verification.
- **Key invalidation mid-conversation**: simulate an invalid/revoked
  key after a successful exchange; confirm the failure shows inline
  on that turn only and prior chat history is preserved (§4.10).
- **Audio input idempotency**: record once, trigger an unrelated
  rerun (e.g. click "Start New Chat" or switch tasks and back), and
  confirm the same recording is not silently reprocessed (§4.4).
- **Audio output autoplay**: `st.audio(..., autoplay=True)` inserted
  via a Streamlit rerun (rather than in the same event-loop tick as
  the triggering click) is known to autoplay inconsistently across
  browsers; verify actual behavior in at least Chrome and Firefox and
  fall back to a visible play button if autoplay can't be relied on.
- **Cross-platform**: run locally on Linux (your dev machine) with
  `streamlit run src/app.py` from the repo root; verify image paths
  and imports resolve. macOS/Windows are validated by construction
  (no OS-specific calls remain anywhere in the codebase after this
  change) rather than by owning three machines to test on.
- **Deployment smoke test**: push to a Streamlit Community Cloud app,
  confirm successful build under the resource ceiling and that the
  app boots to the empty-key state.

## 7. Explicitly Out of Scope

- Rewriting the task-selection UI into a different navigation paradigm
  (tabs, multi-page app) — current structure is kept and polished.
- Adding new AI capabilities beyond what exists today.
- Persisting chat history or uploaded files across sessions/reloads.
- Rate limiting or usage metering of the user's own keys.
