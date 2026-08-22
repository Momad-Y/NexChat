import os


def test_init_llm_model_passes_key_as_argument_not_env(monkeypatch):
    from nlp.RAG import init_llm_model

    captured = {}

    class FakeChatModel:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("nlp.RAG.ChatGoogleGenerativeAI", FakeChatModel)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    init_llm_model("test-gemini-key")

    assert captured["google_api_key"] == "test-gemini-key"
    assert "GOOGLE_API_KEY" not in os.environ


def test_init_embeddings_model_passes_key_as_argument(monkeypatch):
    from nlp.RAG import init_embeddings_model

    captured = {}

    class FakeEmbeddings:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("nlp.RAG.GoogleGenerativeAIEmbeddings", FakeEmbeddings)

    init_embeddings_model("test-gemini-key")

    assert captured["google_api_key"] == "test-gemini-key"
    assert captured["model"] == "models/text-embedding-004"


def test_build_chat_history_accumulates_all_turns():
    from nlp.RAG import build_chat_history
    from langchain_core.messages import HumanMessage, AIMessage

    messages = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "second answer"},
    ]

    history = build_chat_history(messages)

    assert len(history) == 4
    assert isinstance(history[0], HumanMessage) and history[0].content == "first question"
    assert isinstance(history[1], AIMessage) and history[1].content == "first answer"
    assert isinstance(history[2], HumanMessage) and history[2].content == "second question"
    assert isinstance(history[3], AIMessage) and history[3].content == "second answer"


def test_build_chat_history_handles_trailing_unanswered_user_message():
    from nlp.RAG import build_chat_history

    messages = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "unanswered question"},
    ]

    history = build_chat_history(messages)

    assert len(history) == 3


def test_build_chat_history_handles_empty_messages():
    from nlp.RAG import build_chat_history

    assert build_chat_history([]) == []


def test_qa_streams_incremental_chunks():
    from nlp.RAG import qa

    class FakeStreamingModel:
        def stream(self, inputs):
            yield {"answer": "Hello"}
            yield {"answer": " world"}

    chunks = list(qa("a question", FakeStreamingModel(), []))

    assert chunks == ["Hello", " world"]


def test_qa_yields_error_message_when_stream_cannot_start():
    from nlp.RAG import qa

    class FakeBrokenModel:
        def stream(self, inputs):
            raise RuntimeError("invalid key")

    chunks = list(qa("a question", FakeBrokenModel(), []))

    assert chunks == ["An error occurred while generating the answer."]


def test_qa_yields_interruption_message_mid_stream():
    from nlp.RAG import qa

    class FakeInterruptedModel:
        def stream(self, inputs):
            yield {"answer": "Partial"}
            raise RuntimeError("connection dropped")

    chunks = list(qa("a question", FakeInterruptedModel(), []))

    assert chunks[0] == "Partial"
    assert "interrupted" in chunks[-1].lower()


def test_qa_yields_error_message_when_stream_produces_nothing():
    from nlp.RAG import qa

    class FakeEmptyModel:
        def stream(self, inputs):
            return iter([])

    chunks = list(qa("a question", FakeEmptyModel(), []))

    assert chunks == ["An error occurred while generating the answer."]
