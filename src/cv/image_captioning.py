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
    except Exception:
        return "An error occurred while generating the caption."

    return caption.capitalize().strip() + "."
