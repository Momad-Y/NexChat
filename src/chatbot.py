import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
from nlp.question_answering import answer_question
from nlp.summarization import summarize_text
from cv.image_captioning import caption_image
from audio.audio_input import get_audio_input
from audio.audio_output import speak_text
from memory import ChatMemory
from utils import read_pdf, read_csv, read_arxiv, get_file_extension

# Initialize chat memory
chat_memory = ChatMemory()

# Streamlit app
st.title("AI-Powered Chatbot with Multimodal Capabilities")

# Display chat history
context = ""
for message in chat_memory.get_history():
    if message["sender"] == "User":
        st.write(f"**You:** {message['message']}")
        context += f"User: {message['message']} "
    else:
        st.write(f"**Bot:** {message['message']}")
        context += f"Bot: {message['message']} "

# Button to start a new chat
if st.button("Start New Chat"):
    chat_memory.clear_history()
    st.experimental_set_query_params()

# User input
user_input = st.text_input("You: ")

# File upload
uploaded_file = st.file_uploader("Upload a file", type=["pdf", "csv", "jpg", "jpeg", "png"])

# Process user input
if user_input:
    # Add user input to chat history
    chat_memory.add_message("User", user_input)
    
    # Determine the type of task
    if user_input.startswith("summarize"):
        if uploaded_file:
            file_extension = get_file_extension(uploaded_file.name)
            if file_extension == "pdf":
                file_text = read_pdf(uploaded_file)
            elif file_extension == "csv":
                file_text = read_csv(uploaded_file)
            elif file_extension == "arxiv":
                file_text = read_arxiv(uploaded_file)
            else:
                file_text = "Unsupported file type."
            response = summarize_text(file_text)
        else:
            response = summarize_text(user_input)
    elif user_input.startswith("caption"):
        if uploaded_file:
            response = caption_image(uploaded_file)
        else:
            response = "Please upload an image file for captioning."
    else:
        response = answer_question(user_input, context)

    # Add response to chat history
    chat_memory.add_message("Bot", response)
    
    # Display response
    st.write(f"**Bot:** {response}")

    # Follow-up question input
    follow_up_input = st.text_input("Follow-up question:")
    if follow_up_input:
        chat_memory.add_message("User", follow_up_input)
        context += f"User: {follow_up_input} "
        response = answer_question(follow_up_input, context)
        chat_memory.add_message("Bot", response)
        context += f"Bot: {response} "
        st.write(f"**Bot:** {response}")

    # Optional: Audio output
    if st.checkbox("Enable audio output"):
        speak_text(response)

# Optional: Audio input
if st.checkbox("Enable audio input"):
    audio_input = get_audio_input()
    if audio_input:
        st.write(f"**You (audio):** {audio_input}")
        chat_memory.add_message("User", audio_input)
        # Process audio input similarly to text input
        response = answer_question(audio_input, context)
        chat_memory.add_message("Bot", response)
        st.write(f"**Bot:** {response}")
        if st.checkbox("Enable audio output"):
            speak_text(response)