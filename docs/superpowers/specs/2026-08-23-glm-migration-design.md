# NexChat: Replace Gemini with GLM-4.7-Flash (Zero-Cost Stack)

**Status:** Draft for review
**Date:** 2026-08-23

## 1. Goal

Remove Gemini entirely from NexChat and replace it with GLM-4.7-Flash
(Zhipu AI / Z.ai), so the app can be run end-to-end with zero-cost,
no-credit-card-required API keys. Gemini's free tier still requires a
Google Cloud billing account; GLM-4.7-Flash is free with no comparable
gate. This is a pure provider swap — no new user-facing features, no
change to RAG/summarization/captioning behavior beyond what's forced by
the swap itself.

## 2. What Changes and Why

Gemini currently serves two roles in this app: the RAG chat model
(`ChatGoogleGenerativeAI`) and RAG embeddings (`GoogleGenerativeAIEmbeddings`).
GLM has no embeddings API at all (confirmed by reading Z.ai's own docs
and pricing pages directly — the pricing page lists text, vision, image,
video, and audio models, no embedding models). Zhipu's own `embedding-3`
model exists but is paid and is only documented against `bigmodel.cn`
(Zhipu's China-facing platform), not the international `z.ai` endpoint
this app's chat key would use — a separate, unverified account
relationship, not a clean single-provider story.

So this is a two-provider swap, not a one-for-one substitution:

| Feature | Before | After |
|---|---|---|
| RAG chat (answer generation) | Gemini, `ChatGoogleGenerativeAI` | GLM-4.7-Flash, `ChatOpenAI` (OpenAI-compatible) |
| RAG embeddings | Gemini, `GoogleGenerativeAIEmbeddings` | HuggingFace hosted, `HuggingFaceEndpointEmbeddings` |
| Summarization / captioning | HuggingFace | unchanged |
| STT / TTS | Google Web Speech / gTTS (keyless) | unchanged |

BYOK stays a two-key setup — **GLM key + HuggingFace key** — the same
count as before, just one key swapped for another. The HuggingFace key
goes from covering two features (summarization, captioning) to three
(adding embeddings), with no new key type introduced.

## 3. Key Decisions

### 3.1 Chat integration: `ChatOpenAI` (`langchain-openai`) vs. hand-rolled HTTP vs. a GLM-specific SDK

**Option 1 — `langchain_openai.ChatOpenAI` with a `base_url` override.**
For: GLM's API is OpenAI-compatible (`https://api.z.ai/api/paas/v4/`,
standard bearer auth), and `langchain-openai` is the official,
first-party LangChain package for exactly this pattern. `ChatOpenAI`
implements the same `BaseChatModel` interface `ChatGoogleGenerativeAI`
did, so it satisfies `qa()`'s `.stream()` call and LangChain's
`create_history_aware_retriever`/`create_retrieval_chain`/
`create_stuff_documents_chain` chain-construction functions with zero
changes anywhere else in `RAG.py`. Against: adds one new dependency the
codebase hasn't used before.

**Option 2 — hand-rolled `requests.post` calls**, matching
`summarization.py`/`image_captioning.py`'s existing style. For:
consistent with the app's two existing HuggingFace integrations, no new
dependency. Against: the RAG chain (`create_history_aware_retriever` +
`create_retrieval_chain` + `create_stuff_documents_chain`) is built
entirely on LangChain's `Runnable`/chain abstractions — history-aware
query reformulation, document-stuffing into the prompt, and incremental
streaming are all handled by those functions today. Hand-rolling the
chat call would mean reimplementing that entire chain by hand, an
enormous, unjustified scope increase to avoid one small package.

**Option 3 — a GLM-specific SDK or community LangChain integration.**
For: might expose GLM-specific features (e.g. its "thinking" reasoning
mode) beyond generic OpenAI compatibility. Against: community-maintained
and less battle-tested than `langchain-openai`, and this app doesn't
need any GLM-specific capability beyond chat + streaming — extra surface
area with no corresponding need.

**Verdict: Option 1.** The RAG chain's architecture already requires
any chat model to satisfy LangChain's `BaseChatModel` interface;
`langchain-openai` provides that via an official, well-tested package
with a one-line `base_url` change, while hand-rolling would require
rebuilding the whole chain and a GLM-specific SDK adds risk for no
needed capability.

### 3.2 Embeddings integration: `HuggingFaceEndpointEmbeddings` vs. hand-rolled HTTP vs. Zhipu `embedding-3`

**Option 1 — `langchain_huggingface.HuggingFaceEndpointEmbeddings`.**
For: implements LangChain's `Embeddings` interface, so
`create_vector_store`'s `FAISS.from_texts(texts, embedding_model)` and
`create_qa_model`'s `.as_retriever()` need zero changes — this is the
exact same interface `GoogleGenerativeAIEmbeddings` already satisfied.
Confirmed via a clean-venv install that `langchain-huggingface`'s only
hard dependencies are `huggingface-hub`, `langchain-core`, and
`tokenizers` — no torch, no sentence-transformers (those only get
imported if the *local* `HuggingFaceEmbeddings` class is instantiated,
which this design does not use). Against: relies on the class
constructing the correct, currently-live HF router URL internally —
the same kind of endpoint-drift risk that already hit the summarization/
captioning HF integration once (spec 2026-08-22, §4.3) and needs the
same live-key verification before shipping. A second, distinct risk
found by red-team review and confirmed by reading `huggingface_hub`'s
actual source (`get_provider_helper` in
`huggingface_hub/inference/_providers/__init__.py`): leaving `provider`
unset resolves to `provider="auto"`, which triggers a *live network
call* to resolve which marketplace provider currently serves the model
for `feature-extraction` — a failure mode independent of endpoint drift
(the model could be unregistered with every auto-eligible provider even
if the classic `hf-inference` provider itself serves it fine). See
§3.8 — this is fixed by pinning `provider="hf-inference"` explicitly,
not left as residual risk.

**Option 2 — hand-rolled `requests.post` to HF's feature-extraction
endpoint**, wrapped in a small custom class implementing LangChain's
`Embeddings` interface. For: full control, consistent hand-rolled style
with the app's other HF integrations. Against: more code to write and
test for no functional gain over an already-installed, already-verified
first-party library that does exactly this.

**Option 3 — Zhipu's own `embedding-3`.** For: true single-provider
story (chat + embeddings both from Zhipu). Against: paid (not free —
directly against this migration's zero-cost goal), documented only
against `bigmodel.cn` rather than the `z.ai` international endpoint the
chat key uses (unverified whether the same account/key even works
there), and its only LangChain integration is community-maintained
(`langchain_community.embeddings.ZhipuAIEmbeddings`, a separate
`zhipuai` PyPI package) rather than first-party.

**Verdict: Option 1.** Same reasoning as chat: the app already consumes
embeddings via LangChain's `Embeddings` interface, so the first-party
class requires no downstream changes, and it's confirmed free and
torch-free. Option 3 is ruled out on cost and platform-uncertainty
grounds alone — it would compromise the migration's actual goal.

### 3.3 Which embedding model: `BAAI/bge-small-en-v1.5` vs. `sentence-transformers/all-MiniLM-L6-v2`

**Option 1 — `BAAI/bge-small-en-v1.5`.** For: this is the exact model
the pre-migration version of this app used locally, before the original
Gemini migration moved embeddings hosted — known-good fit for this
app's retrieval quality, already validated by its own history. Against:
needs live verification that it's actually served on HF's current
Inference Providers router (not guaranteed by popularity alone).

**Option 2 — `sentence-transformers/all-MiniLM-L6-v2`.** For: the most
widely used default embedding model in LangChain's own documentation
and examples, very likely to be reliably hosted. Against: modestly
lower retrieval quality than bge-small on most benchmarks, and no
history with this specific app.

**Verdict: Option 1, with Option 2 as the documented fallback** if live
verification shows `BAAI/bge-small-en-v1.5` isn't currently served.
Deciding factor: it's a known-good fit with this app's own history, and
the hosting-availability risk is the same *kind* of risk already
handled elsewhere in this plan via a pre-implementation verification
step — not a reason to default to a less-proven model.

### 3.4 BYOK naming: rename to GLM-specific vs. generic provider-agnostic

**Option 1 — rename throughout** (`GLM_KEY_NAME`, `glm_api_key`, "GLM
API key" sidebar label, `GLM_API_KEY` env var). For: matches this
codebase's own existing convention exactly — the other key is literally
named/labeled "HuggingFace," not something generic like
`chat_api_key`; a stale "Gemini" label in a UI that no longer uses
Gemini at all is actively misleading to a user trying to figure out
which key to paste in. Against: a larger, more mechanical diff, since
the name threads through `credentials.py`, `RAG.py`, `vector_cache.py`,
`app.py`, tests, and docs.

**Option 2 — generic naming** (`CHAT_API_KEY`, `chat_api_key`, "Chat
model API key"). For: no rename needed if the provider changes again.
Against: breaks the codebase's established provider-specific naming
pattern, over-engineers for a hypothetical future swap nobody has asked
for, and is less helpful to a user who needs to know specifically to go
get a Z.ai/GLM key.

**Verdict: Option 1.** The existing convention is provider-specific
naming; a stale "Gemini" label is a real, immediate cost, while generic
naming pays for a hypothetical future that YAGNI says to ignore.

### 3.5 `init_RAG`'s two-key signature

**Option 1 — two explicit string parameters**,
`init_RAG(glm_api_key: str, huggingface_api_key: str) -> tuple`. For:
simple, explicit, the natural extension of the existing one-key
pattern; the caller (`app.py`'s QA branch) already has both keys in
scope by the time it calls this. No meaningful against.

**Option 2 — a single credentials dict/dataclass parameter.** For:
marginally more extensible if a third key were ever needed. Against:
over-engineering for two keys, introduces a structure with no current
precedent or benefit — every other function in this codebase takes
plain string key parameters directly.

**Verdict: Option 1.** No alternative serves the current requirement
better; Option 2's only advantage is for a need that doesn't exist yet.

### 3.6 Vector-cache fingerprint: which key to fold in

**No alternatives — forced by fact.** The fingerprint's purpose
(established when the final review caught and fixed a cross-user-style
cache bug in the original migration) is to invalidate the cache exactly
when the credential that determines embedding identity changes.
Embeddings now come from the HuggingFace key, not the GLM key — so
`compute_files_fingerprint` must fold in the HuggingFace key instead of
the GLM key to preserve that property. Folding in the GLM key instead
would silently reintroduce the exact bug the earlier fix closed, just
keyed to the wrong credential.

### 3.7 Dependencies

Add: `langchain-openai` (new — chat). Add: `langchain-huggingface`
(previously removed in the original migration when embeddings moved to
Gemini; re-added now for hosted embeddings — confirmed torch-free for
the class this design uses, §3.2). Remove: `langchain-google-genai` and
its transitive Google API tree (`google-ai-generativelanguage`,
`google-api-core`, `google-api-python-client`, `google-auth`,
`google-auth-httplib2`, `google-generativeai`,
`googleapis-common-protos`) — no longer used anywhere once Gemini is
fully removed. Net effect is a further size reduction on top of the
torch removal from the original migration, not just a lateral swap.

### 3.8 `HuggingFaceEndpointEmbeddings`'s `provider` parameter (added after red-team review)

**No alternatives worth weighing — one option is simply correct.**
Verified directly by reading `huggingface_hub`'s source: leaving
`provider` unset does not silently default to the classic
`hf-inference` endpoint — it resolves to `provider="auto"`, which
fetches the current Inference Providers marketplace mapping for the
model over the network before the embedding call even happens. If
`BAAI/bge-small-en-v1.5` isn't registered with any auto-eligible
provider (plausible — many pre-marketplace sentence-transformers models
were dropped when HF moved off the legacy free API), this fails even
though the classic `hf-inference` provider explicitly still lists
`feature-extraction` as a supported task. Passing `provider="hf-inference"`
explicitly in `init_embeddings_model` collapses this second failure
surface into the one endpoint-drift risk §3.2/§5 already plan to
verify with a live key, instead of stacking two independent unverified
risks.

### 3.9 `create_vector_store` error handling in `app.py` (added after red-team review)

**Option 1 — leave unwrapped, as it is today.** For: no code change,
matches current behavior. Against: an embeddings failure (auth, the
`provider="auto"` marketplace-resolution failure mode this migration
was going to introduce before §3.8's fix, or a genuine network error)
surfaces as a raw Streamlit traceback — the one place in the QA flow
without the friendly-error-message discipline `qa()`, `summarize_text`,
and `caption_image` all already have. This gap is pre-existing (the
same was true for Gemini embeddings), but this migration's changes
increase the odds of hitting it by introducing a new provider-resolution
call in the chain.

**Option 2 — wrap the call and show a friendly message on failure.**
For: closes the one inconsistency in this app's otherwise-universal
"never crash, always show a clear message" pattern, at the cost of a
few lines around an existing call site. Against: none meaningful — this
is a small, self-contained, low-risk addition directly in the file this
migration already touches.

**Verdict: Option 2.** Cheap, consistent with the app's established
error-handling philosophy everywhere else, and made more load-bearing
by this migration's own new failure surface — not scope creep, since
it's the same file and the same call site the migration is already
rewriting.

### 3.10 Simultaneous vs. sequential missing-key messaging (added after red-team review)

**Option 1 — sequential (`if not glm_key: ... elif not hf_key: ...`),
as originally drafted.** For: minimal code, matches the single-message
pattern every other task branch already uses. Against: this is the
*first* feature needing two keys at once — a first-time user with
neither key fills in GLM, reruns, sees "HuggingFace key missing" only
then. A whack-a-mole onboarding flow that didn't exist before this
migration.

**Option 2 — show every missing key at once.** For: a user with zero
keys learns everything they need in one screen instead of two round
trips; costs nothing beyond replacing `elif` with a second `if`.
Against: none meaningful.

**Verdict: Option 2.** Free to fix, and directly serves the migration's
own "make onboarding easy" spirit — no reason to ship the worse of two
equally-cheap options.

## 4. Architecture

### 4.1 `src/nlp/RAG.py`

```python
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpointEmbeddings

GLM_BASE_URL = "https://api.z.ai/api/paas/v4/"
GLM_MODEL = "glm-4.7-flash"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def init_llm_model(glm_api_key: str) -> ChatOpenAI:
    """Initializes the GLM-4.7-Flash chat model via its OpenAI-compatible API."""
    return ChatOpenAI(
        model=GLM_MODEL,
        api_key=glm_api_key,
        base_url=GLM_BASE_URL,
        temperature=0.1,
        max_retries=2,
        timeout=60,
    )


def init_embeddings_model(huggingface_api_key: str) -> HuggingFaceEndpointEmbeddings:
    """Initializes hosted HuggingFace embeddings — no local model, no torch."""
    return HuggingFaceEndpointEmbeddings(
        model=EMBEDDING_MODEL,
        provider="hf-inference",
        task="feature-extraction",
        huggingfacehub_api_token=huggingface_api_key,
    )


def init_RAG(glm_api_key: str, huggingface_api_key: str) -> tuple:
    """Initializes the models and templates for a given user's GLM and HuggingFace keys."""
    llm = init_llm_model(glm_api_key)
    embedding_model = init_embeddings_model(huggingface_api_key)
    prompt, contextualize_q_prompt = init_prompt()

    return llm, embedding_model, prompt, contextualize_q_prompt
```

Both constructors above are verified directly against the actual
installed package signatures (not just documentation) — `ChatOpenAI`'s
`api_key`/`base_url` are confirmed pydantic aliases for
`openai_api_key`/`openai_api_base`, and `HuggingFaceEndpointEmbeddings`'
`huggingfacehub_api_token`/`task`/`provider` are confirmed real fields,
with `task` defaulting to `"feature-extraction"` already (set explicitly
above anyway, for clarity against future default changes) but `provider`
defaulting to `None` — which, per §3.8, must be overridden to
`"hf-inference"` explicitly, since leaving it `None` triggers a live
"auto" marketplace-resolution call with its own independent failure
mode. `timeout=60` on `ChatOpenAI` (found by red-team review, not present in
the original Gemini code) is a deliberate hardening addition: `timeout`
defaults to `None`, which `langchain-openai` passes straight through to
`httpx.Client(timeout=None)` — genuinely unbounded, not a sane library
default — and GLM's free tier publishes no fixed rate limits (Z.ai's own
docs describe throttling as dynamic/concurrency-based rather than a
documented RPM/TPM number), making an unbounded wait against a
possibly-throttled free endpoint a real risk of hanging a Streamlit
worker with only the spinner as feedback. This was already true of the
original Gemini code (which also passed `timeout=None`) but is worth
fixing now while this function is already being rewritten.

No other function in `RAG.py` changes — `create_vector_store`,
`create_qa_model`, `build_chat_history`, `qa()`, and `init_prompt` are
all model-agnostic and already consume `llm`/`embedding_model` purely
through LangChain's `BaseChatModel`/`Embeddings` interfaces. Type
annotations on `create_vector_store`/`create_qa_model` that currently
say `GoogleGenerativeAIEmbeddings`/`ChatGoogleGenerativeAI` update to
`HuggingFaceEndpointEmbeddings`/`ChatOpenAI` (annotation-only, matching
the precedent set the first time this exact situation came up — spec
2026-08-22, Task 5).

### 4.2 `src/credentials.py`

`GEMINI_KEY_NAME = "gemini_api_key"` → `GLM_KEY_NAME = "glm_api_key"`.
`get_gemini_key` → `get_glm_key`. Sidebar label "Gemini API key" → "GLM
API key" (Z.ai)". `.env` var read from `GEMINI_API_KEY` → `GLM_API_KEY`.
Status caption "Gemini connected"/"Gemini key missing" → "GLM
connected"/"GLM key missing". No other change to the module's shape —
`init_credential_state`, `resolve_initial_key`, `missing_key_message`,
`load_dotenv_values` are all provider-agnostic already.

### 4.3 `src/nlp/vector_cache.py`

`compute_files_fingerprint(uploaded_files: list, gemini_api_key: str)`
→ `compute_files_fingerprint(uploaded_files: list, huggingface_api_key:
str)` — signature rename only; the length-prefixed hashing body is
unchanged (§3.6).

### 4.4 `src/app.py`

The Question Answering branch currently gates on one key
(`get_gemini_key`) and calls `init_RAG(gemini_key)` and
`compute_files_fingerprint(uploaded_files, gemini_key)`. It now gates
on **both** keys being present, showing every missing key at once
(§3.10) rather than one at a time:

```python
glm_key = get_glm_key(st.session_state)
hf_key = get_huggingface_key(st.session_state)

if not glm_key:
    st.info(missing_key_message("Question Answering", "GLM"))
if not hf_key:
    st.info(missing_key_message("Question Answering", "HuggingFace"))

if glm_key and hf_key:
    llm, embedding_model, prompt_template, contextualize_q_prompt = init_RAG(glm_key, hf_key)
    ...
    fingerprint = compute_files_fingerprint(uploaded_files, hf_key)
    vector_store = get_cached_vector_store(st.session_state, fingerprint)
    if vector_store is None:
        try:
            vector_store = create_vector_store(uploaded_files, embedding_model)
            store_vector_store(st.session_state, fingerprint, vector_store)
        except Exception:
            st.error("Couldn't index the uploaded files — check your HuggingFace key and try again.")
            vector_store = None
    ...
```

`missing_key_message`'s existing signature (`task_label, provider_label
-> str`) needs no change — it already takes an arbitrary provider label
string, so `"GLM"` and `"HuggingFace"` both work with zero changes to
that function. The `create_vector_store` wrapping (§3.9) is new — the
current code calls it unguarded; on failure `vector_store` stays `None`
so the existing "upload a file to start" downstream messaging naturally
applies instead of a crash.

## 5. Testing Plan

- **Unit tests** (mirroring the pattern from the original migration's
  Task 5): `init_llm_model`/`init_embeddings_model` tests using
  `monkeypatch` to fake `ChatOpenAI`/`HuggingFaceEndpointEmbeddings` and
  assert the key reaches the constructor as an argument, never via
  `os.environ`. `compute_files_fingerprint`'s existing test suite
  updates its second-argument variable name only (still testing the
  same hashing behavior, now conceptually "the embeddings key" instead
  of "the chat key" — no behavior change, so no new test cases needed
  beyond the rename).
- **Pre-implementation verification, before other work depends on it**
  (same discipline as the original migration's §4.3 HF-endpoint
  precedent): confirm with a real GLM key that `glm-4.7-flash` is a
  valid model string against `https://api.z.ai/api/paas/v4/`, and with
  a real HuggingFace key that `BAAI/bge-small-en-v1.5` returns a valid
  embedding vector via `HuggingFaceEndpointEmbeddings`. If either fails,
  the documented fallback is: for the model string, check Z.ai's
  current model list for the closest free-tier equivalent; for
  embeddings, fall back to `sentence-transformers/all-MiniLM-L6-v2`
  (§3.3).
- **Missing-key behavior**: app boots with zero keys, QA branch shows
  **both** missing-key messages at once (§3.10) — not a crash, not just
  the first one; entering only one key leaves exactly the other
  message showing. This is new behavior beyond the original single-key
  gate and needs explicit coverage.
- **`create_vector_store` failure handling** (§3.9): a monkeypatched
  `create_vector_store` that raises must result in a friendly
  `st.error(...)` message and `vector_store` staying `None`, not an
  uncaught exception reaching the user.
- **`init_embeddings_model` passes `provider="hf-inference"`** (§3.8):
  a `monkeypatch`-based test asserting the constructor call includes
  `provider="hf-inference"`, not left unset — this is the one-line fix
  the red-team review's highest-leverage finding depends on, so it gets
  its own explicit assertion rather than relying on the live-key check
  alone to catch a regression.
- **Regression**: full existing test suite (48 tests as of the prior
  migration) must continue passing with only the renamed/retyped
  fixtures updated — no unrelated behavior change.

**Known, documented risk not mitigated by new mechanism:** Z.ai
publishes no fixed rate-limit numbers for GLM-4.7-Flash's free tier —
its docs describe throttling as dynamic/concurrency-based rather than a
stated RPM/TPM quota, and third-party aggregators report inconsistent
daily-request-cap figures. A throttled or rate-limited response during
`qa()`'s two-call pattern (query reformulation + answer generation, both
inside one `qa_model.stream()` invocation) is caught by the same
generic `except Exception` that already handles any other mid-stream
failure (`RAG.py`'s `qa()`, spec 2026-08-22 §4.5) — the user sees the
existing interrupted-response message, not a specific "you're being
rate-limited" explanation. Building rate-limit-specific messaging is
out of scope for this migration (no documented limit to design around,
and the existing generic error handling already prevents a crash) —
noted here as an accepted, pre-existing-pattern risk, not a defect to
fix.

## 6. Explicitly Out of Scope

- Any GLM-specific feature beyond basic chat + streaming (reasoning
  mode, function calling, vision).
- Changing summarization, captioning, STT, or TTS — all HuggingFace/
  Google-Web-Speech/gTTS, all unchanged.
- Repo-level "open source" polish (CONTRIBUTING.md, contributor docs) —
  confirmed with the user this migration's zero-cost-to-run goal is the
  entire scope of "open source" here.
- Supporting both Gemini and GLM as selectable providers — this is a
  hard replacement, not an added option.
