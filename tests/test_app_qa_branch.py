from pathlib import Path

from streamlit.testing.v1 import AppTest

# AppTest.from_file() resolves relative paths against the *calling test file's*
# directory, not the CWD — so build an absolute path off this file instead.
APP_PATH = str(Path(__file__).resolve().parent.parent / "src" / "app.py")


def test_qa_branch_shows_friendly_error_when_indexing_fails(monkeypatch):
    import nlp
    import credentials

    monkeypatch.setattr(credentials, "get_glm_key", lambda: "test-glm-key")
    monkeypatch.setattr(credentials, "get_huggingface_key", lambda: "test-hf-key")

    def failing_create_vector_store(*args, **kwargs):
        raise RuntimeError("simulated embeddings failure")

    # app.py does `from nlp import create_vector_store`, and nlp/__init__.py
    # does `from .RAG import create_vector_store` — so the name app.py resolves
    # is the `nlp` package attribute, NOT `nlp.RAG`'s. Patching nlp.RAG here
    # would leave the real network-calling function in place (verified: it runs
    # and the test times out). Patch the `nlp` package attribute instead.
    monkeypatch.setattr(nlp, "create_vector_store", failing_create_vector_store)

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.sidebar.selectbox[0].select("Question Answering").run()
    at.file_uploader[0].set_value(("notes.txt", b"hello world", "text/plain"))
    at.run()

    errors = [el.value for el in at.error]
    assert any("index" in e.lower() for e in errors), f"no indexing error message in {errors}"
    assert not at.exception
