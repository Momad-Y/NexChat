# This file initializes the NLP module, allowing for easier imports of the NLP functionalities.
from .RAG import (
    init_RAG,
    create_vector_store,
    create_qa_model,
    qa,
    build_chat_history,
    is_qa_failure,
)
from .summarization import summarize_text
