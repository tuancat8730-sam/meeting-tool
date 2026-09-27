import pytest

from meeting_tool.config import MissingApiKeyError, Settings, provider_of, require_api_key


@pytest.mark.parametrize(
    ("model", "provider"),
    [
        ("gemini/gemini-2.5-flash", "gemini"),
        ("anthropic/claude-sonnet-5", "anthropic"),
        ("gpt-4o", "openai"),
    ],
)
def test_provider_of(model: str, provider: str) -> None:
    assert provider_of(model) == provider


def test_require_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(MissingApiKeyError, match="GEMINI_API_KEY"):
        require_api_key("gemini/gemini-2.5-flash")


def test_require_api_key_accepts_alternative_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    require_api_key("gemini/gemini-2.5-flash")


def test_require_api_key_unknown_provider_is_noop() -> None:
    require_api_key("someprovider/model-x")


def test_settings_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.transcribe_model == "gemini/gemini-2.5-flash"
    assert settings.chunk_minutes == 10.0


def test_settings_rejects_overlap_longer_than_chunk() -> None:
    with pytest.raises(ValueError, match="overlap_seconds"):
        Settings(_env_file=None, chunk_minutes=1, overlap_seconds=60)
