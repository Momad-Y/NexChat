import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st

import random
import itertools
import time

from cv import caption_image
from nlp.question_answering import answer_question
from nlp.summarization import summarize_text
from audio import get_audio_input, speak_text, init
from memory import ChatMemory
from utils import read_pdf, read_csv, read_arxiv, get_file_extension


def mygenerator():
    for i in range(10):
        yield f"Hello {i}\n"
        time.sleep(0.5)


# Initialize chat memory
# chat_memory = ChatMemory()

# Initialize audio output engine
# init()
# speak_text("Welcome to the AI-powered chatbot.")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []


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

        # Display the uploaded image in the first column
        col1.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        # Display the caption button in the second column
        if col2.button("Caption Image"):
            response = caption_image(uploaded_file)
            col2.write(f"**Caption:** {response}")
    else:
        st.write("Please upload an image file for captioning.")

else:
    uploaded_file = st.file_uploader(
        "Upload a file", type=["pdf", "csv"], accept_multiple_files=True
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
                generator = mygenerator()

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


# # Display chat history
# context = ""
# for message in chat_memory.get_history():
#     if message["sender"] == "User":
#         st.write(f"**You:** {message['message']}")
#         context += f"User: {message['message']} "
#     else:
#         st.write(f"**Bot:** {message['message']}")
#         context += f"Bot: {message['message']} "

# # Button to start a new chat
# if st.button("Start New Chat"):
#     chat_memory.clear_history()
#     st.experimental_set_query_params()

# # User input
# user_input = st.text_input("You: ")


# # Process user input
# if user_input:
#     # Add user input to chat history
#     chat_memory.add_message("User", user_input)

#     # Determine the type of task
#     if user_input.startswith("summarize"):
#         if uploaded_file:
#             file_extension = get_file_extension(uploaded_file.name)
#             if file_extension == "pdf":
#                 file_text = read_pdf(uploaded_file)
#             elif file_extension == "csv":
#                 file_text = read_csv(uploaded_file)
#             elif file_extension == "arxiv":
#                 file_text = read_arxiv(uploaded_file)
#             else:
#                 file_text = "Unsupported file type."
#             response = summarize_text(file_text)
#         else:
#             response = summarize_text(user_input)
#     elif user_input.startswith("caption"):
#         if uploaded_file:
#             response = caption_image(uploaded_file)
#         else:
#             response = "Please upload an image file for captioning."
#     else:
#         response = answer_question(user_input, context)

#     # Add response to chat history
#     chat_memory.add_message("Bot", response)

#     # Display response
#     st.write(f"**Bot:** {response}")

#     # Follow-up question input
#     follow_up_input = st.text_input("Follow-up question:")
#     if follow_up_input:
#         chat_memory.add_message("User", follow_up_input)
#         context += f"User: {follow_up_input} "
#         response = answer_question(follow_up_input, context)
#         chat_memory.add_message("Bot", response)
#         context += f"Bot: {response} "
#         st.write(f"**Bot:** {response}")

#     # Optional: Audio output
#     if st.checkbox("Enable audio output"):
#         speak_text(response)
# # Optional: Audio input
# if st.checkbox("Enable audio input"):
#     audio_input = get_audio_input()
#     if audio_input:
#         st.write(f"**You (audio):** {audio_input}")
#         chat_memory.add_message("User", audio_input)
#         # Process audio input similarly to text input
#         response = answer_question(audio_input, context)
#         chat_memory.add_message("Bot", response)
#         st.write(f"**Bot:** {response}")
#         if st.checkbox("Enable audio output"):
#             speak_text(response)
