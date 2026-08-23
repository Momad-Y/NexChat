import base64

import requests
from streamlit.runtime.uploaded_file_manager import UploadedFile

API_URL = "https://api.z.ai/api/paas/v4/chat/completions"
MODEL = "glm-4.6v-flash"
DEFAULT_MIME_TYPE = "image/jpeg"
PROMPT = "Describe this image in one short, plain caption (no more than one sentence)."


def caption_image(uploaded_file: UploadedFile, glm_api_key: str) -> str:
    """Captions an image via GLM-4.6V-Flash (Z.ai) — the HuggingFace-hosted
    captioning models this used to call have no live inference provider
    left; GLM's free vision-flash model replaces them."""
    mime_type = getattr(uploaded_file, "type", None) or DEFAULT_MIME_TYPE
    image_b64 = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")

    headers = {"Authorization": f"Bearer {glm_api_key}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
                    },
                ],
            }
        ],
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        caption = response.json()["choices"][0]["message"]["content"]
    except Exception:
        return "An error occurred while generating the caption."

    return caption.strip().capitalize().rstrip(".") + "."
