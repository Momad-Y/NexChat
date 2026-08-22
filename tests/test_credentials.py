from credentials import resolve_initial_key, missing_key_message, init_credential_state


def test_resolve_initial_key_prefers_existing_session_value():
    assert resolve_initial_key("session-key", "dotenv-key") == "session-key"


def test_resolve_initial_key_falls_back_to_dotenv():
    assert resolve_initial_key(None, "dotenv-key") == "dotenv-key"


def test_resolve_initial_key_falls_back_to_empty_string():
    assert resolve_initial_key(None, None) == ""


def test_missing_key_message_names_task_and_provider():
    msg = missing_key_message("Image Captioning", "HuggingFace")
    assert "HuggingFace" in msg
    assert "Image Captioning" in msg


def test_init_credential_state_seeds_missing_keys_only():
    session_state = {"gemini_api_key": "already-set"}
    init_credential_state(session_state, env_values={"HUGGINGFACE_API_KEY": "from-env"})
    assert session_state["gemini_api_key"] == "already-set"
    assert session_state["huggingface_api_key"] == "from-env"


def test_init_credential_state_defaults_to_empty_without_dotenv():
    session_state = {}
    init_credential_state(session_state, env_values={})
    assert session_state["gemini_api_key"] == ""
    assert session_state["huggingface_api_key"] == ""


def test_load_dotenv_values_reads_from_repo_root(monkeypatch, tmp_path):
    import credentials

    fake_env = tmp_path / ".env"
    fake_env.write_text("GEMINI_API_KEY=from-repo-root-env\n")
    monkeypatch.setattr(credentials, "REPO_ROOT", tmp_path)

    values = credentials.load_dotenv_values()

    assert values.get("GEMINI_API_KEY") == "from-repo-root-env"


def test_load_dotenv_values_returns_empty_when_no_env_file(monkeypatch, tmp_path):
    import credentials

    monkeypatch.setattr(credentials, "REPO_ROOT", tmp_path)

    assert credentials.load_dotenv_values() == {}
