# NexChat — Streamlit Community Cloud + BYOK Redesign

**Date:** 2026-08-17
**Status:** Draft for review

## 1. Problem

NexChat works only on the original authors' machines. It cannot deploy to
Streamlit Community Cloud, and two of its three features are already broken
in production. The causes are structural, not cosmetic:

- Credentials are read at **module import time** from a local `.env`, so the
  app cannot accept user-supplied keys and hard-crashes without them.
- Audio capture and playback run **server-side** (PyAudio microphone, pygame
  speaker), which cannot work in a headless container.
- RAG embeddings run **locally** via sentence-transformers, dragging in
  `torch` (~800MB) and likely exceeding Cloud's ~1GB resource ceiling.
- Summarization and captioning call `api-inference.huggingface.co`, which is
  **decommissioned**.

Goal: every existing feature works identically when run locally on Linux,
macOS, and Windows, and when deployed to Community Cloud, with each user
supplying their own API keys.

## 2. Verified findings

Evidence gathered from the current tree (commit `0fc9581`).

| # | Finding | Evidence |
|---|---------|----------|
| F1 | Legacy HF endpoint is dead | `POST api-inference.huggingface.co` → HTTP `000`; `router.huggingface.co` → `401` |
| F2 | Keys read at import | `RAG.py:23-30`, `summarization.py:12-16`, `image_captioning.py:11-15` |
| F3 | RAG initialised at import, every rerun | `app.py:19` `init_RAG()` at module top level |
| F4 | Vector store rebuilt every rerun | `app.py:172` `create_vector_store(...)` uncached |
| F5 | torch pulled in only by embeddings | no `import torch` in `src/`; `RAG.py:64` `HuggingFaceEmbeddings(BAAI/bge-small-en-v1.5)` |
| F6 | Server-side mic | `audio_input.py:18` `sr.Microphone()` |
| F7 | Server-side speaker, blocking | `audio_output.py:16-28` `pygame.mixer` + `while get_busy()` spin |
| F8 | `duration` parameter ignored | `audio_input.py:20` records `DURATION` (5s), not the argument; `app.py:102` asks for 30s |
| F9 | Streaming is fake | `itertools.tee` + full drain before `st.write_stream` at `app.py:106,128,151,200,245` |
| F10 | Artificial latency | `summarization.py:63` `time.sleep(0.5)` per sentence |
| F11 | CWD-dependent paths | `app.py:37,47` `./imgs/icon.png`, `./imgs/logo.png` |
| F12 | CWD-dependent imports | `app.py:11-14` bare `from utils import ...` requires repo-root CWD |
| F13 | Stale repo link | `app.py:57` points at GitLab; project is on GitHub |

## 3. Decisions

### D1 — Embeddings: hosted, drop torch

**For:** Removes `torch`, `sentence-transformers`, and their transitive tree —
the deploy drops from ~1GB to well under 200MB and cold start goes from
minutes to seconds. Reuses the Gemini key BYOK already collects, so it adds
no new credential. Makes the Cloud resource ceiling a non-issue.

**Against:** Embedding now costs the user quota and requires network; large
uploads incur per-chunk API latency that local embedding did not. Loses
offline capability.

**Verdict:** Hosted, via `GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")`.
Deciding factor: no other option reliably fits Community Cloud's ceiling, and
this one costs zero additional keys.

**Alternatives weighed:** CPU-only torch wheel (still 300-500MB, may still
exceed ceiling); environment-conditional local/hosted split (doubles code
paths and makes local and deployed behave differently, contradicting the
stated goal).

### D2 — BYOK scope: Gemini + HuggingFace

**For:** Matches the providers already in use, so summarization keeps BART and
captioning keeps BLIP. Two clearly-labelled inputs is a comprehensible UX.

**Against:** Keeps a second provider that can fail independently, and leaves
speech-to-text on an unauthenticated, rate-limited Google endpoint.

**Verdict:** Two keys. Deciding factor: consolidating onto Gemini would silently
change which models produce user-visible output, which exceeds the mandate to
preserve existing functionality.

### D3 — Key persistence: session-only + local `.env` autofill

**For:** Keys live in `st.session_state`, never written to disk, never logged,
gone when the tab closes — the correct posture for a public deployment. Local
`.env` autofill preserves the maintainer's dev loop.

**Against:** Public users re-enter keys each session.

**Verdict:** As stated. Deciding factor: browser-persisted credentials on a
public multi-tenant app is a security downside with no proportionate benefit.

### D4 — Audio: rebuild in-browser

**For:** `st.audio_input` captures from the *visitor's* microphone via the
browser; `st.audio` plays back in the browser. Identical behaviour local and
deployed. Deletes PyAudio, portaudio, pygame, chime, and the whole
`packages.txt` system-dependency layer — which is also the largest source of
cross-platform breakage.

**Against:** `st.audio_input` requires a recent Streamlit; loses the chime
audio cues that signalled recording start/stop.

**Verdict:** Rebuild in-browser. Deciding factor: it is the only option under
which "works locally on every OS and deployed" is literally true.

**Note:** `SpeechRecognition` is retained but used via `sr.AudioFile` on the
uploaded WAV bytes instead of `sr.Microphone()`. This needs no PortAudio and
runs headless, so STT stays keyless with no system packages.

### D5 — Entry point: `streamlit_app.py` at repo root, `src/` as a package

**For:** Community Cloud looks for `streamlit_app.py` by default. A root entry
point plus absolute `from src.… import …` imports makes the app runnable from
any working directory, fixing F12 permanently.

**Against:** Changes the documented run command; `README.md` needs updating.

**Verdict:** Adopt. Deciding factor: F12 has no robust fix that preserves
CWD-relative imports.

### D6 — Asset paths via `pathlib`

Forced by fact — F11 has no alternative that survives an arbitrary CWD.
Resolve against `Path(__file__).resolve().parent`.

### D7 — Caching strategy (REVISED after review)

The original draft proposed `st.cache_resource` for provider clients and
`st.cache_data` for the vector store. Review found this unsafe as written.

**The problem.** `st.cache_resource` and `st.cache_data` are **global across
every user and session** of a deployed Streamlit app — not per-session. Three
consequences the draft missed:

1. **Unbounded memory growth.** Every distinct upload adds a FAISS index to a
   global cache that is never evicted. On Community Cloud's constrained
   container this is an OOM crash, not a slowdown — and it is triggered by
   normal multi-visitor use, not abuse.
2. **Documents outlive their session.** A content-addressed vector store keeps
   one visitor's embedded document resident in server memory after they leave.
   Content-addressing means another visitor only hits it by uploading a
   byte-identical file, so this is not disclosure — but it is unbounded
   retention of user data with no lifecycle, which is the wrong default for a
   public app.
3. **Shared clients across users.** A globally cached provider client keyed by
   hashed API key is only safe while the hash is the sole key component. Any
   future change that widens the key risks one visitor's client — and quota —
   serving another.

**Verdict:** Vector stores live in `st.session_state`, not a global cache.
They are naturally per-session, die with the session, and cannot accumulate.
`@st.cache_data` is used only for pure, bounded, non-user-specific work
(parsed file text), always with explicit `ttl` and `max_entries`. Provider
clients are constructed per-session and held in `st.session_state`; they are
cheap objects wrapping HTTP calls, so caching them globally buys little and
risks credential-scoped state crossing users.

**Against:** A user re-uploading the same file within one session re-embeds
unless we also memoise per-session by content hash — we will, inside
`st.session_state`, keyed by file hash. Cross-session reuse is deliberately
given up in exchange for bounded memory and clean data lifecycle.

**Deciding factor:** F3 and F4 must be fixed, but the fix cannot introduce
shared global state on a multi-tenant public deployment. Session-scoped state
fixes the recompute problem without inheriting the global-cache problems.

## 4. Architecture

```
streamlit_app.py            # entry point; page config, routing
src/
  config/
    credentials.py          # key resolution: session_state → .env → None
    paths.py                # pathlib asset resolution
  providers/
    gemini.py               # lazy chat + embedding clients
    huggingface.py          # router.huggingface.co calls
  nlp/
    rag.py                  # vector store, retrieval, qa
    summarization.py        # HF router summarization
  cv/
    captioning.py           # HF router captioning
  audio/
    stt.py                  # WAV bytes → text (sr.AudioFile)
    tts.py                  # text → mp3 bytes (gTTS)
  ui/
    sidebar.py              # BYOK key entry, task selection, about
    views/                  # one module per task view
  utils.py                  # file readers
```

**Credential flow.** `credentials.py` exposes `get_key(name)` resolving in
order: `st.session_state` → local `.env` (only when not deployed) → `None`.
Every provider call takes the key as an argument. No module reads a key at
import. A feature whose key is missing renders a clear prompt to enter it
rather than raising.

**Data flow (RAG).** upload → `read_file` → chunk → hashed cache lookup →
hosted embeddings → FAISS → retriever → Gemini chat → true token stream.

## 5. Feature parity matrix

| Feature | Today (local) | Today (deployed) | After |
|---|---|---|---|
| Q&A over documents | works | broken (no key, torch) | works both |
| Text summarization (text) | broken (F1) | broken | works both |
| Text summarization (file) | broken (F1) | broken | works both |
| Text summarization (audio) | broken (F1+F8) | impossible (F6) | works both |
| Image captioning | broken (F1) | broken | works both |
| Audio input | local mic only | impossible | browser mic, both |
| Audio output | local speaker only | impossible | browser player, both |
| Chat history / new chat | works | works | works |

No feature is dropped.

## 6. Dependencies

**Removed:** `torch`, `sentence-transformers`, `langchain-huggingface`,
`PyAudio`, `pygame`, `chime`, plus their transitive tree.
**`packages.txt`:** emptied — no system libraries remain.
**Retained:** `streamlit`, `langchain*`, `faiss-cpu`, `langchain-google-genai`,
`SpeechRecognition`, `gTTS`, `requests`, `PyPDF2`, `pandas`, `python-dotenv`.
**Target:** `requirements.txt` from 119 lines to roughly 15 direct pins.

## 7. UI/UX rework

- **Sidebar:** BYOK key entry at top (`type="password"`), each with a link to
  where the key is obtained and a live validity indicator. Task selection
  below. Author/repo block last, with the GitLab link corrected to GitHub (F13).
- **Gated actions:** feature buttons disable with an explanatory caption when
  the required key is absent, instead of failing mid-call.
- **Real streaming:** remove `itertools.tee` double-consumption (F9) and the
  `time.sleep(0.5)` (F10); accumulate into session state from the single
  consumed stream so tokens appear as generated.
- **Audio:** replace "Start Recording" with `st.audio_input`; replace the
  "Audio Output" button with an `st.audio` player rendered beside responses.
- **Errors:** every provider call returns a typed result; failures render as
  `st.error` with an actionable message, never a silent `except: pass`
  (as at `summarization.py:42` and `audio_output.py:30`).
- **Consistency:** one shared page-header component; consistent spinners and
  empty states across all three views.

## 8. Error handling

Replace bare `except:` blocks with specific handling. Every network call gets
an explicit timeout and distinguishes: missing key (prompt), auth failure
(401/403 → "check your key"), rate limit (429 → "retry shortly"), model
loading (503 → retry with backoff), and transport failure. User-facing copy
never includes the key or raw tracebacks.

## 9. Testing

Unit tests for `credentials.py` resolution order, `utils.read_file` per file
type, chunking, and error mapping — provider calls mocked, no network in CI.
Integration tests behind a marker requiring live keys. Manual verification
matrix across Linux/macOS/Windows and the deployed app, covering every row of
§5. Target 80%+ on non-UI modules.

## 10. Out of scope

Auth/multi-user accounts, persistent conversation storage, streaming TTS,
switching chat providers, and rewriting the summarization/captioning models.

## 11. Verification log

Claims checked empirically rather than assumed (venv, 2026-08-18):

| Claim | Result |
|---|---|
| `import speech_recognition` requires PyAudio | **False** — imports clean with no pyaudio; `pyaudio` not in `sys.modules`; `AudioFile` and `recognize_google` present; only `sr.Microphone()` fails. D4 confirmed. |
| `st.audio_input` availability | **Present in streamlit 1.41.1**, the version already pinned. No upgrade required. |
| Legacy HF endpoint status | **Dead** — `api-inference.huggingface.co` → HTTP `000`; `router.huggingface.co` → `401`. |
| `st.cache_*` scope | **Global across users and sessions.** Drove the D7 revision above. |

Still unverified — must be confirmed during implementation, not assumed:

- Current Gemini embedding model id, and per-request batch/rate limits when
  embedding large uploads through `langchain-google-genai`.
- Streamlit Community Cloud's exact memory ceiling, and measured install size
  after the dependency cull.
- Whether `recognize_google` accepts `st.audio_input` WAV output without
  resampling (sample-rate/encoding assumptions).
- Upload size limits, FAISS memory profile, and session-state growth under
  sustained use.
