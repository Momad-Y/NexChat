# NexChat — Streamlit Community Cloud + BYOK Redesign

**Date:** 2026-08-19
**Status:** Draft for review

## 1. Problem

NexChat works only on the original authors' Windows machines with their
API keys baked in. It cannot deploy to Streamlit Community Cloud, and
two of its features are broken in production today.

Verified findings from the audit:

| # | Finding | Evidence |
|---|---------|----------|
| F1 | HuggingFace endpoint is dead | `api-inference.huggingface.co` returns HTTP `000`; `router.huggingface.co` returns `401`. Summarization and captioning are broken now. |
| F2 | Credentials read at import time | `RAG.py:23-29`, `summarization.py:12-16`, `image_captioning.py:11-15` |
| F3 | Audio capture is server-side | `audio_input.py:18` opens `sr.Microphone()` — no device exists on Cloud |
| F4 | Audio playback is server-side and blocking | `audio_output.py:16-28` spins on `pygame.mixer.music.get_busy()` |
| F5 | `torch` (~800MB) pulled in only for embeddings | `RAG.py:64` `HuggingFaceEmbeddings`; torch is imported nowhere in `src/` |
| F6 | Vector store rebuilt on every rerun | `app.py:172`, no caching — re-embeds all files per interaction |
| F7 | Streaming is fake | `itertools.tee` + full drain before `st.write_stream` (`app.py:106,128,151,200,245`) |
| F8 | Artificial latency | `summarization.py:63` `time.sleep(0.5)` per sentence |
| F9 | CWD-dependent paths and imports | `app.py:37,47` `./imgs/...`; `from utils import read_file` |
| F10 | `duration` parameter ignored | `audio_input.py:20` uses module `DURATION`; 30s request records 5s |
| F11 | Stale repo link | `app.py:57` points at GitLab |

## 2. Goals

- Deploy on Streamlit Community Cloud within its 1GB resource limit.
- BYOK: each user supplies their own Gemini and HuggingFace keys.
- Identical behaviour local (Linux/macOS/Windows) and deployed.
- **No feature dropped.** Every capability survives, including both audio features.
- Improved UI/UX.

## Non-goals

- No new features beyond what exists today.
- No auth, accounts, or persistence across sessions.
- Not changing the RAG approach (FAISS + history-aware retriever stays).

## 3. Decisions

Each decision below was weighed against alternatives; the deciding factor is stated.

### D1 — Embeddings move to a hosted API

**Chosen:** Google `text-embedding-004` via `langchain-google-genai`, reusing the Gemini key.

- **For:** Removes `torch`, `sentence-transformers`, and `langchain-huggingface` — the only reason the install approaches 1GB. Cold start drops from minutes to seconds. Reuses a key BYOK already collects, so no third credential.
- **Against:** Embedding now costs network latency and consumes the user's Gemini quota; it no longer works offline.
- **Deciding factor:** F5 makes this the difference between deploying and not deploying. Nothing else on the list is a hard blocker.
- **Alternatives:** CPU-only torch wheel (still 300-500MB, may still exceed the cap); environment-conditional local/hosted split (doubles code paths and breaks the parity goal).

### D2 — BYOK covers Gemini + HuggingFace

**Chosen:** Two sidebar key inputs. STT remains on the keyless Google Web Speech endpoint.

- **For:** Matches the providers already in use; smallest rewrite; keeps BART and BLIP.
- **Against:** Retains one unauthenticated, rate-limited dependency (STT) that we cannot control.
- **Deciding factor:** Consolidating everything onto Gemini would mean rewriting summarization and captioning to different models, changing output quality in ways the user has not asked for.

### D3 — Keys are session-scoped, with local `.env` autofill

**Chosen:** `st.session_state` only; a local `.env`, when present, prefills the inputs.

- **For:** Nothing is written to disk or logged. Safe for a public deployment while keeping the local dev loop fast.
- **Against:** Keys must be re-entered after a browser reload.
- **Deciding factor:** Browser-persisted credentials on a public app is a security downside with no offsetting benefit for the primary use case.

### D4 — Audio moves into the browser

**Chosen:** `st.audio_input` for capture, `st.audio` for playback. `streamlit==1.41.1` already supports both.

- **For:** The only option where "works locally on every OS and deployed" is literally true. Deletes PyAudio, pygame, chime, portaudio, and the entire `packages.txt`. Fixes F3, F4, F10 at once.
- **Against:** Recording UX changes — the user clicks record/stop in the browser instead of a fixed-duration server capture. gTTS output becomes a play button rather than autoplay.
- **Deciding factor:** Server-side audio cannot work on Cloud at all; there is no device to open.

### D5 — Real streaming replaces the tee-and-drain pattern

**Chosen:** Stream directly to `st.write_stream`, accumulating into session state via the generator itself.

- **For:** Fixes F7 and F8. Users see tokens as they arrive instead of waiting for the full response and then watching a replay.
- **Against:** Requires each call site to capture the final text differently (a small wrapper generator that appends as it yields).
- **Deciding factor:** The current pattern makes the app feel slower than it is, and the fix is contained.

### D6 — Path and import hygiene

**Chosen:** Anchor asset paths to `Path(__file__).parent`, and make `src` a proper package with relative imports.

- **For:** The app runs from any working directory on any OS. Fixes F9 and the `streamlit run src/app.py` fragility.
- **Against:** Touches every import statement.
- **Deciding factor:** Cross-platform correctness is an explicit goal.

## 4. Architecture

### 4.1 Credential layer (new)

`src/config.py` — the single place credentials are resolved.

```
get_keys() -> Keys            # reads st.session_state, falls back to .env locally
require(provider) -> str      # raises a typed error if the key is missing
```

Rules:
- No module reads a key at import time. All reads happen inside functions, per call.
- Missing keys surface as a friendly in-app prompt, never a stack trace.
- Keys are never logged, echoed, or written to disk.

### 4.2 Provider clients (rewritten)

`src/providers/` replaces the scattered `requests.post` calls.

- `huggingface.py` — points at `https://router.huggingface.co/hf-inference/models/...` (fixes F1). Handles 401 (bad key), 429 (rate limit), 503 (model loading) distinctly.
- `gemini.py` — lazily constructs `ChatGoogleGenerativeAI` and the embedding model per call, using the session key.

### 4.3 RAG changes

- `init_RAG()` at module scope is deleted. Models are built lazily once keys exist.
- The vector store is memoised in **`st.session_state`**, keyed on a SHA-256 of the
  uploaded file bytes. It rebuilds only when the uploaded set changes, fixing F6.

> **SECURITY — do not use `st.cache_resource` or `st.cache_data` here.**
> Streamlit's own source describes these caches as *"shared across all users,
> sessions, and reruns"*
> (`streamlit/runtime/caching/cache_resource_api.py`). On Community Cloud the
> process is shared by every visitor, so a globally cached vector store would
> serve one user's uploaded documents to another user, and a globally cached
> provider client would bill one user's BYOK key for another user's requests.
>
> Rule for this codebase: **anything derived from a user's API key or uploaded
> content lives in `st.session_state` only.** The global caches may be used
> only for genuinely static, non-user-derived data.

An earlier draft of this spec specified `st.cache_resource` for the vector
store. That was a document-disclosure bug and is retracted.

### 4.4 Audio (rewritten)

- `audio/input.py` — accepts the `UploadedFile` from `st.audio_input`, transcodes in-memory, sends to `SpeechRecognition`. No microphone, no chime.
- `audio/output.py` — returns gTTS MP3 **bytes**. The caller renders `st.audio(bytes)`. Never blocks.

### 4.5 UI/UX rework

- Sidebar: key entry (password-masked) with live validation status, then task selection, then credits. Repo link corrected to GitHub (F11).
- Clear empty/disabled states — features that need a key say so instead of erroring.
- Consistent two-column result layout across all three tasks.
- Replace the three duplicated "Audio Output" buttons with one shared result component.
- Keep the balloons.

## 5. Feature parity matrix

Every row must work in both columns before this is done.

| Feature | Local (Linux/macOS/Win) | Streamlit Cloud | Change required |
|---|---|---|---|
| Q&A over uploaded docs | ✅ | ✅ | Lazy keys, cached vector store, hosted embeddings |
| Multi-file upload (pdf/csv/txt/md) | ✅ | ✅ | None |
| ArXiv document read | ✅ | ✅ | None |
| Chat history + "Start New Chat" | ✅ | ✅ | None |
| Text summarization — text input | ✅ | ✅ | HF router endpoint, real streaming |
| Text summarization — file input | ✅ | ✅ | HF router endpoint, real streaming |
| Text summarization — audio input | ✅ | ✅ | `st.audio_input` (was server mic) |
| Image captioning | ✅ | ✅ | HF router endpoint |
| Voice input → Q&A | ✅ | ✅ | `st.audio_input` (was server mic) |
| Spoken output (all 3 tasks) | ✅ | ✅ | `st.audio` bytes (was pygame blocking) |
| Balloons easter egg | ✅ | ✅ | None |

No feature is dropped or environment-gated.

## 6. Dependencies

**Removed:** `torch`, `sentence-transformers`, `langchain-huggingface`, `PyAudio`, `pygame`, `chime`.
**`packages.txt`:** becomes empty — no system packages needed.
**Expected result:** ~1GB → well under 200MB, comfortably inside the Cloud limit.

## 7. Error handling

Boundaries that must fail gracefully with a user-facing message:

- Missing/invalid API key → sidebar prompt, feature disabled, no traceback.
- HF 503 (model cold) → retry with backoff, then a clear "model is loading" message.
- HF 429 / Google quota → "rate limited, try again shortly".
- STT failure (no speech, endpoint down) → inline message, chat unaffected.
- Unsupported/corrupt upload → per-file message, other files still processed.
- Network failure → explicit message; never a silent `except: pass`.

The bare `except: pass` in `audio_output.py:30` and the bare `except:` blocks in `summarization.py` and `RAG.py` are replaced with typed handling.

## 8. Testing

- **Unit:** `utils.py` readers (each file type, empty, malformed), `config.py` key resolution precedence, provider error mapping.
- **Integration:** provider clients against mocked HTTP for 200/401/429/503.
- **Manual parity pass:** every row of §5 exercised locally and on a deployed instance.
- Target 80%+ coverage on non-UI modules.

## 9. Phasing

1. **Foundation** — package/import/path fixes (D6), config layer (D1-D3 credential plumbing).
2. **Providers** — HF router migration (F1), hosted embeddings, lazy model init, typed errors.
3. **Audio** — in-browser capture and playback (D4).
4. **Performance** — vector store caching (F6), real streaming (D5, F7, F8).
5. **UI/UX** — sidebar, states, shared result component, GitHub link.
6. **Deploy** — slim requirements, empty packages.txt, deploy and run the parity matrix.

## 10. Verification status

Checked directly against the installed `.venv` (streamlit 1.41.1) and live network,
rather than assumed:

| Claim | Result | Method |
|---|---|---|
| `st.audio_input` exists in streamlit 1.41.1 | ✅ present | `hasattr(st,'audio_input')` |
| `GoogleGenerativeAIEmbeddings` in installed langchain-google-genai | ✅ imports | direct import |
| `st.cache_resource` is cross-user | ⚠️ **confirmed shared** | streamlit source string |
| SpeechRecognition reads WAV without ffmpeg/pydub | ✅ works | parsed a synthetic WAV; `pydub` not installed |
| `langchain_huggingface` confined to RAG.py | ✅ 5 refs, all in `RAG.py` | grep |
| legacy `api-inference.huggingface.co` | ❌ dead (HTTP 000) | curl |
| `router.huggingface.co` | ✅ alive (HTTP 401) | curl |
| Google STT host reachable | ✅ responds (HTTP 411) | curl |
| gTTS host reachable | ✅ responds (HTTP 200) | curl |

Because SpeechRecognition parses WAV natively, **`ffmpeg` is not required** and
`packages.txt` can be emptied as planned (§6).

**Still to confirm at runtime (phase 3):** the exact container `st.audio_input`
returns. The plan assumes WAV, which SpeechRecognition consumes directly. If it
returns WebM/Opus instead, a decode step is needed and `ffmpeg` must return to
`packages.txt` — this is the one assumption that could change §6.

## 11. Open risks

- Hosted embeddings change retrieval quality vs `bge-small-en-v1.5`. Mitigation: verify against the existing sample docs in `data/` during phase 2.
- The keyless Google Web Speech endpoint is undocumented and may rate-limit or disappear. Mitigation: isolate behind the provider interface so a keyed STT provider can be swapped in without touching callers.
- Community Cloud may still be slow on first load. Mitigation: measure after phase 6.
- **Shared egress IP.** Every visitor to the deployed app leaves from the same
  Cloud IP. The keyless STT endpoint and gTTS both rate-limit per IP, so they may
  degrade under concurrent use in a way they never do locally. This is inherent to
  keeping STT keyless (D2) and is the strongest argument for revisiting that
  decision if usage grows. Mitigation: both sit behind the provider interface, so
  a keyed provider can be swapped in without touching callers.
