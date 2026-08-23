from pathlib import Path

from streamlit.testing.v1 import AppTest

# AppTest.from_file() resolves relative paths against the *calling test file's*
# directory, not the CWD — so build an absolute path off this file instead.
APP_PATH = str(Path(__file__).resolve().parent.parent / "src" / "app.py")


def test_missing_keys_shows_single_error_and_stops(monkeypatch):
    import credentials

    monkeypatch.setattr(credentials, "get_glm_key", lambda: "")
    monkeypatch.setattr(credentials, "get_huggingface_key", lambda: "")

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    errors = [el.value for el in at.error]
    assert any("configured" in e.lower() for e in errors), f"no misconfiguration error in {errors}"
    assert not at.exception


def test_missing_only_one_key_still_shows_error(monkeypatch):
    import credentials

    monkeypatch.setattr(credentials, "get_glm_key", lambda: "test-glm-key")
    monkeypatch.setattr(credentials, "get_huggingface_key", lambda: "")

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    errors = [el.value for el in at.error]
    assert any("configured" in e.lower() for e in errors), f"no misconfiguration error in {errors}"
    assert not at.exception


def test_both_keys_present_renders_selected_task_instead_of_error(monkeypatch):
    import credentials

    monkeypatch.setattr(credentials, "get_glm_key", lambda: "test-glm-key")
    monkeypatch.setattr(credentials, "get_huggingface_key", lambda: "test-hf-key")

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    errors = [el.value for el in at.error]
    assert not any("configured" in e.lower() for e in errors), f"unexpected misconfiguration error in {errors}"
    assert not at.exception
