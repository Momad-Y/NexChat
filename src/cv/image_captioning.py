from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Initialize the processor and model for image captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption_image(image_path):
    # Open the image file
    image = Image.open(image_path)
    
    # Preprocess the image
    inputs = processor(image, return_tensors="pt")
    
    # Generate the caption
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    
    return caption