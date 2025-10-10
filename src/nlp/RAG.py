from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable
import streamlit as st

from utils import read_file

from dotenv import dotenv_values, find_dotenv
import os
import time

from streamlit.runtime.uploaded_file_manager import UploadedFile
from typing import Generator

try:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = dotenv_values(find_dotenv())[
        "HUGGINGFACE_API_KEY"
    ]
    os.environ["GOOGLE_API_KEY"] = dotenv_values(find_dotenv())["GEMINI_API_KEY"]
except Exception as e:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACE_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]


def init_llm_model() -> ChatGoogleGenerativeAI:
    """
    Initializes and returns an instance of the ChatGoogleGenerativeAI model.

    Args:
        None

    Returns:
        ChatGoogleGenerativeAI: An instance of the ChatGoogleGenerativeAI model.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.1,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    return llm


def init_embeddings_model() -> HuggingFaceEmbeddings:
    """
    Initializes the Hugging Face embeddings model.

    Args:
        None

    Returns:
        HuggingFaceEmbeddings: The Hugging Face embeddings model.

    """
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    return embedding_model


def init_prompt() -> ChatPromptTemplate:
    """
    Initializes the prompt template.

    Args:
        None

    Returns:
        tuple: A tuple of the initialized prompt templates.
    """
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    system_prompt = """
    You are a helpful assistant tasked with answering questions based on the provided context. Follow these rules carefully:

    1. Only use the provided context to answer the question. Do not add information or assumptions beyond what is given.
    2. If the answer cannot be determined from the context, explicitly state: 
    "I don't know, but you can check other resources online."
    3. If the answer can be determined, provide it concisely, limited to a maximum of five sentences.

    Context:
    {context}

    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    return prompt, contextualize_q_prompt


def init_RAG() -> tuple:
    """
    Initializes the models and templates.

    Args:
        None

    Returns:
        tuple: A tuple of the initialized models and prompt templates.
    """
    llm = init_llm_model()
    embedding_model = init_embeddings_model()
    prompt, contextualize_q_prompt = init_prompt()

    return llm, embedding_model, prompt, contextualize_q_prompt


def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> list:
    """
    Splits the given text into chunks of the specified size.

    Args:
        text (str): The text to be split.
        max_chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The overlap between each chunk.

    Returns:
        list: A list of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    split_texts = text_splitter.split_text(text)

    return split_texts


def create_vector_store(
    uploaded_files: list[UploadedFile], embedding_model: HuggingFaceEmbeddings
) -> FAISS:
    """
    Initializes the vector store.

    Args:
        texts (list): A list of UploadedFiles.
        embedding_model: The embedding model to use for creating the vectors.

    Returns:
        FAISS: A FAISS vector store.
    """
    all_split_texts = []

    for uploaded_file in uploaded_files:
        texts = read_file(uploaded_file)
        split_texts = split_text(texts)
        all_split_texts.extend(split_texts)

    vector_store = FAISS.from_texts(all_split_texts, embedding_model)

    return vector_store


def create_qa_model(
    vector_store: FAISS,
    llm: ChatGoogleGenerativeAI,
    prompt: ChatPromptTemplate,
    contextualize_q_prompt: ChatPromptTemplate,
) -> Runnable:
    """
    Initializes the QA model.

    Args:
        vector_store (FAISS): The vector store.
        llm (ChatGoogleGenerativeAI): The language model.
        prompt (ChatPromptTemplate): The prompt template.
        contextualize_q_prompt (ChatPromptTemplate): The contextualize question prompt template.

    Returns:
        Runnable: A runnable QA model.

    """
    history_aware_retriever = create_history_aware_retriever(
        llm, vector_store.as_retriever(kwargs={"k": 6}), contextualize_q_prompt
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    retrieval_qa = create_retrieval_chain(
        history_aware_retriever, question_answer_chain
    )

    return retrieval_qa


def qa(text: str, qa_model: Runnable, messages: list) -> Generator[str, str, str]:
    """
    Generates answers to questions based on the given text.

    Args:
        text (str): The text to generate answers from.
        qa_model (Runnable): The QA model.
        messages (list): A list of messages.

    Returns:
        Generator[str, str, str]: A generator that yields the generated answers, or None if an error occurred.
    """
    chat_history = []

    for message in messages:
        if message["role"] == "user":
            user_message = HumanMessage(content=message["content"])
        elif message["role"] == "assistant":
            ai_answer = message["content"]

    try:
        chat_history.extend([user_message, ai_answer])
    except Exception as e:
        pass

    try:
        response = qa_model.invoke({"chat_history": chat_history, "input": text})
        answer = response["answer"].strip()
    except Exception as e:
        return None

    words = answer.split(" ")

    for word in words:
        yield word + " "
        time.sleep(0.1)
