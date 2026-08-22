import os


def test_init_llm_model_passes_key_as_argument_not_env(monkeypatch):
    from nlp.RAG import init_llm_model

    captured = {}

    class FakeChatModel:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("nlp.RAG.ChatOpenAI", FakeChatModel)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    init_llm_model("test-glm-key")

    assert captured["api_key"] == "test-glm-key"
    assert captured["base_url"] == "https://api.z.ai/api/paas/v4/"
    assert captured["model"] == "glm-4.7-flash"
    assert "OPENAI_API_KEY" not in os.environ


def test_init_embeddings_model_passes_key_as_argument(monkeypatch):
    from nlp.RAG import init_embeddings_model

    captured = {}

    class FakeEmbeddings:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("nlp.RAG.HuggingFaceEndpointEmbeddings", FakeEmbeddings)

    init_embeddings_model("test-hf-key")

    assert captured["huggingfacehub_api_token"] == "test-hf-key"
    assert captured["model"] == "BAAI/bge-small-en-v1.5"
    assert captured["provider"] == "hf-inference"
    assert captured["task"] == "feature-extraction"


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


def test_qa_yields_error_message_when_messages_are_malformed():
    from nlp.RAG import qa

    class FakeStreamingModel:
        def stream(self, inputs):
            yield {"answer": "should not reach here"}

    malformed_messages = [{"role": "user"}]  # missing "content" key

    chunks = list(qa("a question", FakeStreamingModel(), malformed_messages))

    assert chunks == ["An error occurred while generating the answer."]


def test_is_qa_failure_true_for_error_message():
    from nlp.RAG import is_qa_failure, QA_ERROR_MESSAGE

    assert is_qa_failure(QA_ERROR_MESSAGE) is True


def test_is_qa_failure_true_for_interrupted_response():
    from nlp.RAG import is_qa_failure

    assert is_qa_failure("Here is a partial answer\n\n⚠️ Response interrupted — please try again.") is True


def test_is_qa_failure_false_for_real_answer():
    from nlp.RAG import is_qa_failure

    assert is_qa_failure("The answer to your question is 42.") is False
