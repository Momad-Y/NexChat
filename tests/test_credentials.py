def test_resolve_fixed_key_prefers_secrets_value():
    from credentials import resolve_fixed_key

    assert resolve_fixed_key("secret-key", "dotenv-key") == "secret-key"


def test_resolve_fixed_key_falls_back_to_dotenv():
    from credentials import resolve_fixed_key

    assert resolve_fixed_key(None, "dotenv-key") == "dotenv-key"


def test_resolve_fixed_key_falls_back_to_empty_string():
    from credentials import resolve_fixed_key

    assert resolve_fixed_key(None, None) == ""


def test_resolve_fixed_key_treats_empty_secret_as_missing():
    from credentials import resolve_fixed_key

    assert resolve_fixed_key("", "dotenv-key") == "dotenv-key"


def test_load_dotenv_values_reads_from_repo_root(monkeypatch, tmp_path):
    import credentials

    fake_env = tmp_path / ".env"
    fake_env.write_text("GLM_API_KEY=from-repo-root-env\n")
    monkeypatch.setattr(credentials, "REPO_ROOT", tmp_path)

    values = credentials.load_dotenv_values()

    assert values.get("GLM_API_KEY") == "from-repo-root-env"


def test_load_dotenv_values_returns_empty_when_no_env_file(monkeypatch, tmp_path):
    import credentials

    monkeypatch.setattr(credentials, "REPO_ROOT", tmp_path)

    assert credentials.load_dotenv_values() == {}


def test_get_glm_key_reads_from_secrets_first(monkeypatch):
    import credentials

    monkeypatch.setattr(credentials.st, "secrets", {"GLM_API_KEY": "secret-glm-key"})
    monkeypatch.setattr(credentials, "load_dotenv_values", lambda: {"GLM_API_KEY": "dotenv-glm-key"})

    assert credentials.get_glm_key() == "secret-glm-key"


def test_get_glm_key_falls_back_to_dotenv_when_no_secret(monkeypatch):
    import credentials

    monkeypatch.setattr(credentials.st, "secrets", {})
    monkeypatch.setattr(credentials, "load_dotenv_values", lambda: {"GLM_API_KEY": "dotenv-glm-key"})

    assert credentials.get_glm_key() == "dotenv-glm-key"


def test_get_glm_key_empty_when_unconfigured(monkeypatch):
    import credentials

    monkeypatch.setattr(credentials.st, "secrets", {})
    monkeypatch.setattr(credentials, "load_dotenv_values", lambda: {})

    assert credentials.get_glm_key() == ""


def test_get_huggingface_key_reads_from_secrets_first(monkeypatch):
    import credentials

    monkeypatch.setattr(credentials.st, "secrets", {"HUGGINGFACE_API_KEY": "secret-hf-key"})
    monkeypatch.setattr(
        credentials, "load_dotenv_values", lambda: {"HUGGINGFACE_API_KEY": "dotenv-hf-key"}
    )

    assert credentials.get_huggingface_key() == "secret-hf-key"


def test_get_huggingface_key_falls_back_to_dotenv_when_no_secret(monkeypatch):
    import credentials

    monkeypatch.setattr(credentials.st, "secrets", {})
    monkeypatch.setattr(
        credentials, "load_dotenv_values", lambda: {"HUGGINGFACE_API_KEY": "dotenv-hf-key"}
    )

    assert credentials.get_huggingface_key() == "dotenv-hf-key"


def test_missing_keys_message_names_both_env_vars():
    from credentials import missing_keys_message

    msg = missing_keys_message()
    assert "GLM_API_KEY" in msg
    assert "HUGGINGFACE_API_KEY" in msg
