from pathlib import Path

from streamlit.testing.v1 import AppTest

# AppTest.from_file() resolves relative paths against the *calling test file's*
# directory, not the CWD — so build an absolute path off this file instead.
APP_PATH = str(Path(__file__).resolve().parent.parent / "src" / "app.py")


def test_summarization_file_uploader_does_not_accept_csv(monkeypatch):
    import credentials

    monkeypatch.setattr(credentials, "get_glm_key", lambda: "test-glm-key")
    monkeypatch.setattr(credentials, "get_huggingface_key", lambda: "test-hf-key")

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.sidebar.selectbox[0].select("Text Summarization").run()
    at.radio[0].set_value("File").run()

    allowed = at.file_uploader[0].allowed_type
    assert "csv" not in allowed and ".csv" not in allowed, f"csv still allowed: {allowed}"
    assert not at.exception


def test_qa_file_uploader_still_accepts_csv(monkeypatch):
    # CSV summarization produces garbage on every model tested (not prose),
    # so it was removed from Summarization — but QA goes through the RAG/LLM
    # chain instead of a dedicated summarizer, a different mechanism that
    # wasn't shown to have the same problem, so it keeps CSV support.
    import credentials

    monkeypatch.setattr(credentials, "get_glm_key", lambda: "test-glm-key")
    monkeypatch.setattr(credentials, "get_huggingface_key", lambda: "test-hf-key")

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.sidebar.selectbox[0].select("Question Answering").run()

    allowed = at.file_uploader[0].allowed_type
    assert ".csv" in allowed, f"csv unexpectedly missing: {allowed}"
    assert not at.exception
