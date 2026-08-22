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
