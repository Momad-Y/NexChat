# AI-Powered Chatbot with Multimodal Capabilities

This project is an AI-powered chatbot that supports multimodal capabilities, including text, audio, and image processing. The chatbot can answer questions, summarize text, and caption images.

## Requirements

To install the required dependencies, run:

```sh
pip install -r requirements.txt
Project Structure
css
Copy code
src/
    audio/
        __init__.py
        audio_input.py
        audio_output.py
    cv/
        __init__.py
        image_captioning.py
    nlp/
        __init__.py
        question_answering.py
        summarization.py
    memory.py
    utils.py
    chatbot.py
Usage
To run the Streamlit app, execute the following command:

sh
Copy code
streamlit run src/chatbot.py
Features
Question Answering: Uses NLP to answer questions based on the context.
Text Summarization: Summarizes text from user input or uploaded files.
Image Captioning: Generates captions for uploaded images.
Audio Input/Output: Supports audio input and output for user interactions.
File Descriptions
chatbot.py: Main file to run the Streamlit app.
audio_input.py: Handles audio input.
audio_output.py: Handles audio output.
image_captioning.py: Handles image captioning.
question_answering.py: Handles question answering.
summarization.py: Handles text summarization.
memory.py: Manages chat history.
utils.py: Utility functions for file handling.
License
This project is licensed under the MIT License.

vbnet
Copy code

Feel free to let me know if you need additional customizations!