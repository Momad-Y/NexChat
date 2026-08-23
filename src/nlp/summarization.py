import requests
from typing import Generator

API_URL = "https://router.huggingface.co/hf-inference/models/pszemraj/led-large-book-summary"
MAX_CHUNK_SIZE = 4000
MAX_NEW_TOKENS = 180


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
                response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
                response.raise_for_status()
                yield response.json()[0]["summary_text"] + " "
            except Exception:
                yield "An error occurred while generating the summary."
                return

    else:
        payload = {
            "inputs": text,
            "parameters": {"max_new_tokens": MAX_NEW_TOKENS},
        }
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
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
