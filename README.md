# NexChat (Multimodal AI Chatbot)

## Project Description

NexChat is a multimodal AI chatbot with versatile capabilities, including document-based question answering and summarization, image captioning, and audio interaction.

## Features

- **Question Answering:** Powered by RAG using `glm-4.7-flash` (Z.ai) as the LLM, hosted HuggingFace embeddings (`BAAI/bge-small-en-v1.5`) as the embedding model — no local embedding model, no torch dependency — and Faiss as the vector store.
- **Summarization:** Utilizes `pszemraj/led-large-book-summary` (HuggingFace) for document summarization — a long-context (16K token) model chosen over the previous `facebook/bart-large-cnn` (1024 token limit) to reduce lossy chunking on longer PDF/TXT/MD input.
- **Image Captioning:** Powered by `glm-4.6v-flash` (Z.ai), a free vision-language model — HuggingFace's classic single-purpose captioning models no longer have any live inference provider, so this task runs on GLM instead.
- **Text-to-Speech (TTS):** Uses `gTTS` to convert text responses into spoken audio, played back in the browser through `st.audio`.
- **Speech Recognition:** Recordings are captured in the browser with Streamlit's `st.audio_input` widget and transcribed with Google's `speech_recognition` library — no server-side audio device access.

## Methodology (Tech Stack)

- **Programming Language:** Python 3.11.\*
- **Frameworks and Libraries:**
    - Streamlit for the UI
    - LangChain for retrieval-augmented generation
    - GLM (Z.ai, OpenAI-compatible) for chat (`glm-4.7-flash`, via `langchain-openai`) and image captioning (`glm-4.6v-flash`, via a direct chat-completions call)
    - HuggingFace Inference API (hosted) for embeddings (via `langchain-huggingface`) and summarization
    - Faiss for vector storage and retrieval
    - gTTS for text-to-speech
    - SpeechRecognition for audio-to-text processing

## Project File Structure

```
NexChat/
├── data/
│   ├── test_data...
├── imgs/
│   ├── logos_and_screenshots...
├── src/
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── audio_input.py
│   │   ├── audio_output.py
│   ├── cv/
│   │   ├── __init__.py
│   │   ├── image_captioning.py
│   ├── nlp/
│   │   ├── __init__.py
│   │   ├── RAG.py
│   │   ├── summarization.py
│   │   ├── vector_cache.py
│   ├── app.py
│   ├── credentials.py
│   ├── paths.py
│   ├── rate_limit.py
│   ├── utils.py
├── tests/
│   ├── __init__.py
│   ├── test_app_credentials_gate.py
│   ├── test_app_qa_branch.py
│   ├── test_audio_input.py
│   ├── test_audio_output.py
│   ├── test_credentials.py
│   ├── test_image_captioning.py
│   ├── test_paths.py
│   ├── test_rag.py
│   ├── test_rate_limit.py
│   ├── test_smoke.py
│   ├── test_summarization.py
│   ├── test_utils.py
│   ├── test_vector_cache.py
├── .streamlit/
│   ├── secrets.toml.example   # copy to secrets.toml for deployed keys
├── .env                       # optional, local development only
├── .env.example
├── .gitignore
├── conftest.py
├── packages.txt                # empty — no system packages needed
├── README.md
├── requirements.txt
├── requirements-dev.txt
```

## Setup and Usage

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/Momad-Y/NexChat.git
    cd NexChat
    ```

2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows use `source venv/Scripts/activate`
    ```
3. Install Python dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Configuration

NexChat runs on two free-tier API keys — **GLM (Z.ai)** and **HuggingFace** — configured
once by whoever runs the app, not by each visitor. Both providers currently offer these
for free, so a single deployed instance can serve everyone at no cost to the operator.

- **Local development:** create a `.env` file at the repository root:
    ```
    GLM_API_KEY=your_key_here
    HUGGINGFACE_API_KEY=your_key_here
    ```
- **Deployed (Streamlit Community Cloud):** set the same two keys in the app's
  **Settings → Secrets** panel, using TOML syntax:
    ```toml
    GLM_API_KEY = "your_key_here"
    HUGGINGFACE_API_KEY = "your_key_here"
    ```
    See `.streamlit/secrets.toml.example` for a template — copy it to
    `.streamlit/secrets.toml` for local secrets-based testing (this file is
    gitignored; Community Cloud's Secrets panel is the source of truth once deployed).

If either key is missing, the app shows a single configuration error instead of
starting — this is a deployment issue for the operator to fix, not something an
end user needs to act on.

**Shared-quota rate limiting:** because every visitor draws from the same free-tier
keys, each browser session is capped at 20 requests (combined across all three
features) before it's asked to refresh the page. This bounds how much of the shared
quota any single visitor can consume.

### Running the Application

1. Start the chatbot:
    ```bash
    streamlit run src/app.py
    ```
2. Audio is fully browser-based: recording goes through the browser's microphone via
   Streamlit's `st.audio_input` widget (your browser will ask for microphone access)
   and playback happens in the browser via `st.audio`. The server never touches an
   audio device, so this works identically locally and when deployed.

## Deployment (Streamlit Community Cloud)

NexChat deploys straight from its GitHub repository — no separate build step or
container image needed.

1. Push your changes to [github.com/Momad-Y/NexChat](https://github.com/Momad-Y/NexChat).
2. On [share.streamlit.io](https://share.streamlit.io), create a new app and point it at
   this GitHub repository, branch `main`, with `src/app.py` as the entrypoint.
3. In the app's **Settings → Secrets** panel, add the two keys described in
   [Configuration](#configuration) above (`GLM_API_KEY`, `HUGGINGFACE_API_KEY`).
4. Deploy. Every push to `main` redeploys automatically.

## Demo Screenshots

### Question Answering

![Question Answering](./imgs/question_answering_screenshot.png)

### Summarization

![Summarization](./imgs/summarization_screenshot.png)

### Image Captioning

![Image Captioning](./imgs/image_captioning_screenshot.png)

## Used Resources

- **Hugging Face Inference Providers Documentation:** [https://huggingface.co/docs/inference-providers](https://huggingface.co/docs/inference-providers)
- **Z.ai GLM API Documentation:** [https://docs.z.ai/guides/overview/quick-start](https://docs.z.ai/guides/overview/quick-start)
- **LangChain Documentation:** [https://docs.langchain.com](https://docs.langchain.com)
- **Streamlit Documentation:** [https://docs.streamlit.io](https://docs.streamlit.io)

## Conclusion

NexChat integrates retrieval-augmented question answering, summarization, image
captioning, and browser-based audio into a single interactive chatbot — running
entirely on free-tier GLM and HuggingFace models, with no paid API required to
host or run it. It's fully open source (MIT licensed) and deploys directly from
this GitHub repository to Streamlit Community Cloud, so anyone can fork it, add
their own keys, and have a working instance in minutes.

## [Repository Link](https://github.com/Momad-Y/NexChat)

## Done By

**Begad M Tamim**
[Github](https://github.com/begad-tamim) | [LinkedIn](https://www.linkedin.com/in/begad-tamim/) | [Email](mailto:begadtamim.a@gmail.com)

**Mohamed Y Abdelnasser**
[Github](https://github.com/Momad-Y) | [LinkedIn](https://www.linkedin.com/in/mohamed-y-abdelnasser/) | [Email](mailto:Mohamed.Y.Abdelnasser@gmail.com)
