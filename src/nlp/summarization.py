import requests
from dotenv import dotenv_values, find_dotenv
from typing import Generator
import time

API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
MAX_CHUNK_SIZE = 1000
MAX_NEW_TOKENS = 50

huggingface_api_key = dotenv_values(find_dotenv())["HUGGINGFACE_API_KEY"]
headers = {"Authorization": f"Bearer {huggingface_api_key}"}


def summarize_text(text: str) -> Generator[str, str, str]:
    """
    Summarizes the given text using an AI-powered summarization API.

    Args:
        text (str): The text to be summarized.

    Returns:
        str: The summarized text.
    """
    if len(text) > MAX_CHUNK_SIZE:
        text_chunks = [
            text[i : i + MAX_CHUNK_SIZE] for i in range(0, len(text), MAX_CHUNK_SIZE)
        ]

        for chunk in text_chunks:
            payload = {
                "inputs": chunk,
                "parameters": {"max_new_tokens": MAX_NEW_TOKENS},
            }
            response = requests.post(API_URL, headers=headers, json=payload)
            try:
                yield response.json()[0]["summary_text"] + " "
            except:
                pass

    else:
        payload = {"inputs": text}
        response = requests.post(API_URL, headers=headers, json=payload)
        try:
            summary = response.json()[0]["summary_text"]
        except:
            summary = "An error occurred while generating the summary."

        # split the summary into sentences
        sentences = summary.split(".")

        # Remove empty strings
        sentences = [sentence for sentence in sentences if sentence]

        for sentence in sentences:
            yield sentence.strip() + ". "
            time.sleep(0.5)
