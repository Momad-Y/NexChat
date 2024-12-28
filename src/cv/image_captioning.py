import requests
from dotenv import dotenv_values, find_dotenv
from streamlit.runtime.uploaded_file_manager import UploadedFile

API_URL = (
    "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
)

huggingface_api_key = dotenv_values(find_dotenv())["HUGGINGFACE_API_KEY"]
headers = {"Authorization": f"Bearer {huggingface_api_key}"}


def caption_image(uploaded_file: UploadedFile) -> str:
    """
    Captions an image using an AI-powered model.

    Args:
        uploaded_file (UploadedFile): The image file to be captioned.

    Returns:
        str: The generated caption for the image, or an error message if the captioning process failed.
    """
    data = uploaded_file.getvalue()
    response = requests.post(API_URL, headers=headers, data=data)
    try:
        caption = response.json()[0]["generated_text"]
    except:
        caption = "an error occurred while generating the caption"
    
    return caption.capitalize().strip() + "."
