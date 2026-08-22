import streamlit as st

import random
import time

from audio import get_audio_input, speak_text, has_processed, mark_processed
from cv import caption_image
from nlp import (
    summarize_text,
    init_RAG,
    create_vector_store,
    create_qa_model,
    qa,
    is_qa_failure,
)
from nlp.vector_cache import compute_files_fingerprint, get_cached_vector_store, store_vector_store
from utils import read_file, custom_message_generator
from paths import asset_path
from credentials import render_key_sidebar, get_glm_key, get_huggingface_key, missing_key_message

random.seed(time.time())

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
    hf_key = get_huggingface_key(st.session_state)
    uploaded_files = st.file_uploader(
        "Upload a file", type=["jpg", "jpeg", "png"], accept_multiple_files=False
    )
    st.divider()

    if not hf_key:
        st.info(missing_key_message("Image Captioning", "HuggingFace"))
    elif uploaded_files:
        # Split the page into two columns
        col1, col2 = st.columns(2)
        _, col21, col22 = col2.columns([1, 6, 6])

        # Display the uploaded image in the first column
        col1.image(uploaded_files, caption="Uploaded Image", use_container_width=True)

        # Display the caption button in the second column
        if col21.button("Caption Image"):
            st.session_state.image_caption = caption_image(uploaded_files, hf_key)
        if st.session_state.image_caption:
            col2.write(f"**Caption:** {st.session_state.image_caption}")

        if col22.button("Audio Output", key="audio_image_caption"):
            audio_bytes = speak_text(st.session_state.image_caption)
            if audio_bytes:
                col2.audio(audio_bytes, format="audio/mp3", autoplay=True)
    else:
        st.write("Please upload an image file for captioning.")

elif task_name == "Text Summarization":
    hf_key = get_huggingface_key(st.session_state)
    if not hf_key:
        st.info(missing_key_message("Text Summarization", "HuggingFace"))
    else:
        text_input = st.radio("Select the input type:", ["Text", "File", "Audio"])

        if text_input == "Audio":
            audio_blob = st.audio_input("Record text to summarize")
            if audio_blob and not has_processed(
                st.session_state, audio_blob.getvalue(), key="audio_summarization_last_hash"
            ):
                mark_processed(
                    st.session_state, audio_blob.getvalue(), key="audio_summarization_last_hash"
                )
                recognized = get_audio_input(audio_blob.getvalue())
                if recognized:
                    st.session_state.audio_input = recognized.capitalize().strip() + "."
                    st.write(f"**You (audio):** {st.session_state.audio_input}")
                    st.session_state.text_summarization = st.write_stream(
                        summarize_text(recognized, hf_key)
                    )
                else:
                    st.error("Couldn't understand that recording — please try again.")

        elif text_input == "Text":
            if query := st.text_area("Enter a text for summarization:"):
                query_bytes = query.encode("utf-8")
                if not has_processed(st.session_state, query_bytes, key="text_summarization_last_hash"):
                    mark_processed(st.session_state, query_bytes, key="text_summarization_last_hash")
                    st.session_state.text_summarization = st.write_stream(summarize_text(query, hf_key))
                elif st.session_state.text_summarization:
                    st.markdown(st.session_state.text_summarization)
            else:
                st.write_stream(custom_message_generator("Please enter a text for summarization."))

        else:
            uploaded_files = st.file_uploader(
                "Upload a file", type=["pdf", "csv", "txt", "md"], accept_multiple_files=False
            )
            if uploaded_files:
                text = read_file(uploaded_files)
                if text == "Unsupported file type.":
                    st.write_stream(custom_message_generator(text))
                else:
                    file_bytes = uploaded_files.getvalue()
                    if not has_processed(st.session_state, file_bytes, key="file_summarization_last_hash"):
                        mark_processed(st.session_state, file_bytes, key="file_summarization_last_hash")
                        st.session_state.text_summarization = st.write_stream(summarize_text(text, hf_key))
                    elif st.session_state.text_summarization:
                        st.markdown(st.session_state.text_summarization)
            else:
                st.write_stream(custom_message_generator("Please upload a file for summarization."))

        if st.button("Audio Output", key="audio_text_summarization"):
            summary_text = (
                st.session_state.text_summarization.split("**Summary:**")[-1].strip()
                if st.session_state.text_summarization
                else "No response to output."
            )
            audio_bytes = speak_text(summary_text)
            if audio_bytes:
                st.audio(audio_bytes, format="audio/mp3", autoplay=True)

else:
    glm_key = get_glm_key(st.session_state)
    hf_key = get_huggingface_key(st.session_state)

    if not glm_key:
        st.info(missing_key_message("Question Answering", "GLM"))
    if not hf_key:
        st.info(missing_key_message("Question Answering", "HuggingFace"))

    if glm_key and hf_key:
        llm, embedding_model, prompt_template, contextualize_q_prompt = init_RAG(glm_key, hf_key)

        uploaded_files = st.file_uploader(
            "Upload a file", type=["pdf", "csv", "txt", "md"], accept_multiple_files=True
        )
        st.divider()

        vector_store = qa_model = None

        if uploaded_files:
            fingerprint = compute_files_fingerprint(uploaded_files, hf_key)
            vector_store = get_cached_vector_store(st.session_state, fingerprint)
            if vector_store is None:
                with st.spinner("Indexing files…"):
                    try:
                        vector_store = create_vector_store(uploaded_files, embedding_model)
                        store_vector_store(st.session_state, fingerprint, vector_store)
                    except Exception:
                        st.error("Couldn't index the uploaded files — check your HuggingFace key and try again.")
                        vector_store = None
            else:
                st.caption("Using cached index for this file set.")

            if vector_store is not None:
                qa_model = create_qa_model(vector_store, llm, prompt_template, contextualize_q_prompt)

        # Display the chat interface
        for message in st.session_state.messages:
            avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        # React to user input
        if query := st.chat_input():
            # Display user message in chat message container
            st.chat_message("user", avatar="🧑‍💻").markdown(query)

            # Generate response
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Wait for the response..."):
                    if qa_model:
                        response = st.write_stream(qa(query, qa_model, st.session_state.messages))
                    else:
                        response = st.write_stream(
                            custom_message_generator("Please upload a file to start the chat.")
                        )

            if qa_model and response and not is_qa_failure(response):
                st.session_state.messages.append({"role": "user", "content": query})
                st.session_state.messages.append({"role": "assistant", "content": response})

            # Display a random balloon animation
            if random.random() > 0.9 and response and qa_model and not is_qa_failure(response):
                st.balloons()

        _, col1, col2, col3 = st.columns([1, 3, 3, 3])

        # Add a button to clear the chat history
        if col1.button("Start New Chat"):
            st.session_state.messages = []
            st.rerun()

        # Add a widget for voice questions
        with col2:
            audio_blob = st.audio_input("Ask by voice")
        if audio_blob and not has_processed(
            st.session_state, audio_blob.getvalue(), key="audio_qa_last_hash"
        ):
            mark_processed(st.session_state, audio_blob.getvalue(), key="audio_qa_last_hash")
            recognized = get_audio_input(audio_blob.getvalue())
            if recognized:
                recognized = recognized.capitalize()
                st.chat_message("user", avatar="🧑‍💻").markdown(recognized)
                with st.chat_message("assistant", avatar="🤖"):
                    with st.spinner("Wait for the response..."):
                        if qa_model:
                            response = st.write_stream(qa(recognized, qa_model, st.session_state.messages))
                        else:
                            response = st.write_stream(
                                custom_message_generator("Please upload a file to start the chat.")
                            )
                if qa_model and response and not is_qa_failure(response):
                    st.session_state.messages.append({"role": "user", "content": recognized})
                    st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                st.error("Couldn't understand that recording — please try again.")

        # Add a button for audio output
        if col3.button("Audio Output", key="audio_qa"):
            try:
                last_message = st.session_state.messages[-1]
            except IndexError:
                last_message = {"role": "assistant", "content": "No response to output."}

            audio_response = (
                last_message["content"] if last_message["role"] == "assistant" else "No response to output."
            )
            audio_bytes = speak_text(audio_response)
            if audio_bytes:
                col3.audio(audio_bytes, format="audio/mp3", autoplay=True)
