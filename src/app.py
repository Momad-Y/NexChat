import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st

import random
import itertools
import time

from audio import get_audio_input, speak_text
from cv import caption_image
from nlp import summarize_text, init_RAG, create_vector_store, create_qa_model, qa
from utils import read_file, custom_message_generator
from paths import asset_path
from credentials import render_key_sidebar, get_gemini_key, get_huggingface_key, missing_key_message

random.seed(time.time())

# Initialize the RAG model
llm, embedding_model, prompt_template, contextualize_q_prompt = init_RAG()

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []

if "image_caption" not in st.session_state:
    st.session_state.image_caption = ""

if "text_summarization" not in st.session_state:
    st.session_state.text_summarization = ""
    
if "audio_input" not in st.session_state:
    st.session_state.audio_input = ""

# Set page config
st.set_page_config(
    page_title="NexChat",
    page_icon=str(asset_path("icon.png")),
    layout="centered",
    initial_sidebar_state="expanded",
)

# Streamlit app
st.title("NexChat 🤖")
st.write("### AI Powered Chatbot")

# Display logo
st.sidebar.image(str(asset_path("logo.png")), use_container_width=True)

# Display task selection
task_name = st.sidebar.selectbox(
    "Select the task you would like to perform:",
    ["Question Answering", "Text Summarization", "Image Captioning"],
)

render_key_sidebar()

with st.sidebar.expander("About"):
    st.markdown("## **[Repository Link](https://github.com/Momad-Y/NexChat)**")
    st.markdown("## Done By:")
    st.markdown("##### **Begad M Tamim**")
    st.markdown(
        "##### [Github](https://github.com/begad-tamim) | [LinkedIn](https://www.linkedin.com/in/begad-tamim/) | [Email](mailto:begadtamim.a@gmail.com)"
    )
    st.markdown("##### **Mohamed Y Abdelnasser**")
    st.markdown(
        "##### [Github](https://github.com/Momad-Y) | [LinkedIn](https://www.linkedin.com/in/mohamed-y-abdelnasser/) | [Email](mailto:Mohamed.Y.Abdelnasser@gmail.com)"
    )

if task_name == "Image Captioning":
    uploaded_files = st.file_uploader(
        "Upload a file", type=["jpg", "jpeg", "png"], accept_multiple_files=False
    )
    st.divider()

    if uploaded_files:
        # Split the page into two columns
        col1, col2 = st.columns(2)
        _, col21, col22 = col2.columns([1, 6, 6])

        # Display the uploaded image in the first column
        col1.image(uploaded_files, caption="Uploaded Image", use_container_width=True)

        # Display the caption button in the second column
        if col21.button("Caption Image"):
            response = caption_image(uploaded_files)
            st.session_state.image_caption = response
            col2.write(f"**Caption:** {st.session_state.image_caption}")

        if col22.button("Audio Output", key="audio_image_caption"):
            col2.write(f"**Caption:** {st.session_state.image_caption}")
            speak_text(st.session_state.image_caption)

    else:
        st.write("Please upload an image file for captioning.")

elif task_name == "Text Summarization":
    text_input = st.radio("Select the input type:", ["Text", "File", "Audio"])
    response = ""

    if text_input == "Audio":
        if st.button("Start Recording", use_container_width=True):
            audio_input = get_audio_input(duration=30)
            if audio_input:
                st.session_state.audio_input = audio_input.capitalize().strip() + "."
                generator = summarize_text(audio_input)
                generator, generator2 = itertools.tee(generator)
                for chunk in generator2:
                    response += chunk
                st.session_state.text_summarization = response
            else:
                generator = custom_message_generator("An error occurred while recording the audio.")
            
            st.write(f"**You (audio):** {st.session_state.audio_input}")

        else:
            generator = custom_message_generator("Click the button to start recording.")

        st.write_stream(generator)
        
        if st.button("Audio Output", key="audio_text_summarization"):
            st.write(f"**You (audio):** {st.session_state.audio_input}")
            st.write(st.session_state.text_summarization)
            speak_text(st.session_state.text_summarization.split("**Summary:**")[-1].strip() if st.session_state.text_summarization != "" else "No response to output.")
            
    elif text_input == "Text":
        if query := st.text_area("Enter a text for summarization:"):
            generator = summarize_text(query)
            generator, generator2 = itertools.tee(generator)
            for chunk in generator2:
                response += chunk
            st.session_state.text_summarization = response
        else:
            generator = custom_message_generator("Please enter a text for summarization.")

        st.write_stream(generator)
        
        if st.button("Audio Output", key="audio_text_summarization"):
            speak_text(st.session_state.text_summarization.split("**Summary:**")[-1].strip() if st.session_state.text_summarization != "" else "No response to output.")

    else:
        uploaded_files = st.file_uploader(
            "Upload a file",
            type=["pdf", "csv", "txt", "md"],
            accept_multiple_files=False,
        )
        if uploaded_files:
            text = read_file(uploaded_files)
            if text == "Unsupported file type.":
                generator = custom_message_generator(text)
            generator = summarize_text(text)
            generator, generator2 = itertools.tee(generator)
            for chunk in generator2:
                response += chunk
            st.session_state.text_summarization = response
        else:
            generator = custom_message_generator("Please upload a file for summarization.")

        st.write_stream(generator)
        
        if st.button("Audio Output", key="audio_text_summarization"):
            speak_text(st.session_state.text_summarization.split("**Summary:**")[-1].strip() if st.session_state.text_summarization != "" else "No response to output.")

else:
    uploaded_files = st.file_uploader(
        "Upload a file", type=["pdf", "csv", "txt", "md"], accept_multiple_files=True
    )
    st.divider()

    vector_store = qa_model = None

    if uploaded_files:
        vector_store = create_vector_store(uploaded_files, embedding_model)
        qa_model = create_qa_model(
            vector_store, llm, prompt_template, contextualize_q_prompt
        )

    # Display the chat interface
    for message in st.session_state.messages:
        if message["role"] == "user":
            avatar = "🧑‍💻"
        else:
            avatar = "🤖"

        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # Initialize response
    response = ""

    # React to user input
    if query := st.chat_input():
        # Display user message in chat message container
        st.chat_message("user", avatar="🧑‍💻").markdown(query)

        # Generate response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Wait for the response..."):
                if qa_model:
                    generator = qa(query, qa_model, st.session_state.messages)
                    generator, generator2 = itertools.tee(generator)
                    for chunk in generator2:
                        response += chunk

                    if response:
                        st.session_state.messages.append(
                            {"role": "user", "content": query}
                        )
                        st.session_state.messages.append(
                            {"role": "assistant", "content": response}
                        )

                    else:
                        generator = custom_message_generator(
                            "An error occurred while generating the answer."
                        )

                else:
                    generator = custom_message_generator(
                        "Please upload a file to start the chat."
                    )

            st.write_stream(generator)

        # Display a random balloon animation
        if random.random() > 0.9 and response and qa_model:
            st.balloons()

    _, col1, col2, col3 = st.columns([1, 3, 3, 3])
    
    # Add a button to clear the chat history
    if col1.button("Start New Chat"):
        st.session_state.messages = []
        st.rerun()
        
    # Add a button for audio input
    if col2.button("Audio Input"):
        audio_input = get_audio_input()
        if audio_input:
            audio_input = audio_input.capitalize()
            st.chat_message("user", avatar="🧑‍💻").markdown(audio_input)
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Wait for the response..."):
                    if qa_model:
                        generator = qa(audio_input, qa_model, st.session_state.messages)
                        generator, generator2 = itertools.tee(generator)
                        for chunk in generator2:
                            response += chunk

                        if response:
                            st.session_state.messages.append({"role": "user", "content": audio_input})
                            st.session_state.messages.append({"role": "assistant", "content": response})

                        else:
                            generator = custom_message_generator("An error occurred while generating the answer.")
                    else:
                        generator = custom_message_generator("Please upload a file to start the chat.")
                    
                st.write_stream(generator)
                
    # Add a button for audio output
    if col3.button("Audio Output", key="audio_qa"):
        audio_response = ""
        try:
            last_message = st.session_state.messages[-1]
        except:
            last_message = {"role": "assistant", "content": "No response to output."}
            
        if last_message["role"] == "assistant":
            audio_response = last_message["content"]
        else:
            audio_response = "No response to output."
            
        speak_text(audio_response)
