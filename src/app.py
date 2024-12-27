import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st

import random
import itertools
import time

from audio import get_audio_input, speak_text
from cv import caption_image
from nlp import qa, summarize_text
from utils import read_file


def test_generator():
    for i in range(10):
        yield f"Hello {i}\n"
        time.sleep(0.5)


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "image_caption" not in st.session_state:
    st.session_state.image_caption = ""

if "text_summarization" not in st.session_state:
    st.session_state.text_summarization = ""

# Set page config
st.set_page_config(
    page_title="NexChat",
    page_icon="./imgs/icon.png",
    layout="centered",
    initial_sidebar_state="expanded",
)


# Streamlit app
st.title("NexChat 🤖")
st.write("### AI Powered Chatbot")

# Display logo
st.sidebar.image("./imgs/logo.png", use_container_width=True)

# Display task selection
task_name = st.sidebar.selectbox(
    "Select the task you would like to perform:",
    ["Question Answering", "Text Summarization", "Image Captioning"],
)

# Display the repository link and authors information
st.sidebar.markdown(
    "## **[Repositoriy Link](https://gitlab.com/begad-tamim/ai-powered-chatbot.git)**"
)
st.sidebar.markdown("## Done By:")
st.sidebar.markdown("##### **Begad M Tamim**")
st.sidebar.markdown(
    "##### [Github](https://github.com/begad-tamim) | [LinkedIn](https://www.linkedin.com/in/begad-tamim/) | [Email](mailto:begadtamim.a@gmail.com)"
)
st.sidebar.markdown("##### **Mohamed Y Abdelnasser**")
st.sidebar.markdown(
    "##### [Gitlab](https://gitlab.com/Momad-Y) | [LinkedIn](https://www.linkedin.com/in/mohamed-y-abdelnasser/) | [Email](mailto:Mohamed.Y.Abdelnasser@gmail.com)"
)

if task_name == "Image Captioning":
    uploaded_file = st.file_uploader(
        "Upload a file", type=["jpg", "jpeg", "png"], accept_multiple_files=False
    )
    st.divider()

    if uploaded_file:
        # Split the page into two columns
        col1, col2 = st.columns(2)
        _, col21, col22 = col2.columns([1, 6, 6])

        # Display the uploaded image in the first column
        col1.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        # Display the caption button in the second column
        if col21.button("Caption Image"):
            response = caption_image(uploaded_file)
            response = response.capitalize()
            response += "."
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
        if st.button("Start Recording"):
            audio_input = get_audio_input()
            if audio_input:
                st.write(f"**You (audio):** {audio_input}")
                generator = summarize_text(audio_input)
                generator, generator2 = itertools.tee(generator)
                for chunk in generator2:
                    response += chunk
                st.write_stream(generator)
                st.session_state.text_summarization = response
            else:
                st.write("No audio input detected. Please try again.")

        else:
            st.write("Click the button above to start recording an audio input.")

        if st.button("Audio Output", key="audio_text_summarization"):
            speak_text(st.session_state.text_summarization)

    elif text_input == "Text":
        if prompt := st.text_area("Enter a text for summarization:"):
            generator = summarize_text(prompt)
            generator, generator2 = itertools.tee(generator)
            for chunk in generator2:
                response += chunk
            st.write_stream(generator)
            st.session_state.text_summarization = response
        else:
            st.write("Please enter a text for summarization.")

        if st.button("Audio Output", key="audio_text_summarization"):
            speak_text(st.session_state.text_summarization)

    else:
        uploaded_file = st.file_uploader(
            "Upload a file",
            type=["pdf", "csv", "txt", "md"],
            accept_multiple_files=False,
        )
        if uploaded_file:
            text = read_file(uploaded_file)
            if text == "Unsupported file type.":
                st.write("Unsupported file type. Please upload a supported file type.")
                st.stop()
            generator = summarize_text(text)
            generator, generator2 = itertools.tee(generator)
            for chunk in generator2:
                response += chunk
            st.write_stream(generator)
            st.session_state.text_summarization = response
        else:
            st.write("Please upload a file or enter a text for summarization.")

        if st.button("Audio Output", key="audio_text_summarization"):
            speak_text(st.session_state.text_summarization)


else:
    uploaded_file = st.file_uploader(
        "Upload a file", type=["pdf", "csv", "txt", "md"], accept_multiple_files=True
    )
    st.divider()

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
    if prompt := st.chat_input():
        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message in chat message container
        st.chat_message("user", avatar="🧑‍💻").markdown(prompt)

        # Generate response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Wait for the response..."):
                # generator = generate_response(
                #     vector_store=vector_store,
                #     model=model,
                #     prompt=prompt,
                # )

                # Create a test generator
                generator = test_generator()

                generator, generator2 = itertools.tee(generator)

                st.write_stream(generator)

                for chunk in generator2:
                    response += chunk

        # Add response to session state
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Display a random balloon animation
        random.seed()
        if random.random() > 0.9:
            st.balloons()

    # Add a button to clear the chat history
    if st.button("Start New Chat"):
        st.session_state.messages = []
