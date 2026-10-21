from nextrole.config import Settings


def test_settings_have_safe_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.profile_path is None
    assert settings.llm_provider is None
    assert settings.embedding_provider is None
    assert settings.log_level == "INFO"


def test_settings_read_prefixed_environment(monkeypatch) -> None:
    monkeypatch.setenv("NEXTROLE_LLM_PROVIDER", "test-provider")
    monkeypatch.setenv("LLM_API_KEY", "secret-value")
    monkeypatch.setenv("NEXTROLE_LOG_LEVEL", "DEBUG")

    settings = Settings(_env_file=None)

    assert settings.llm_provider == "test-provider"
    assert settings.llm_api_key is not None
    assert settings.llm_api_key.get_secret_value() == "secret-value"
    assert settings.log_level == "DEBUG"
