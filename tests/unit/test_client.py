from types import SimpleNamespace
from typing import Any

import litellm
import pytest

from meeting_tool.llm.client import LLMClient, LLMOutputError, extract_json
from meeting_tool.models.transcript import RawTranscript

VALID = '{"segments": [{"start": "00:01", "speaker": "A", "text": "Xin chào"}]}'
MESSAGES = [{"role": "user", "content": "hi"}]


def response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


class FakeCompletion:
    def __init__(self, *outcomes: str | Exception) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return response(outcome)


@pytest.fixture(autouse=True)
def _pricing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "completion_cost", lambda **_: 0.002)
    monkeypatch.setattr(litellm, "supports_response_schema", lambda **_: True)


def test_valid_output() -> None:
    fake = FakeCompletion(VALID)
    result = LLMClient("gemini/x", completion_fn=fake).complete_structured(MESSAGES, RawTranscript)
    assert result.value.segments[0].text == "Xin chào"
    assert result.cost_usd == 0.002
    assert (result.prompt_tokens, result.completion_tokens) == (10, 5)
    assert fake.calls[0]["response_format"] is RawTranscript
    assert fake.calls[0]["model"] == "gemini/x"


def test_json_mode_when_schema_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "supports_response_schema", lambda **_: False)
    fake = FakeCompletion(VALID)
    LLMClient("x/y", completion_fn=fake).complete_structured(MESSAGES, RawTranscript)
    assert fake.calls[0]["response_format"] == {"type": "json_object"}


def test_fenced_output_is_accepted() -> None:
    fake = FakeCompletion(f"Đây là kết quả:\n```json\n{VALID}\n```")
    result = LLMClient("gemini/x", completion_fn=fake).complete_structured(MESSAGES, RawTranscript)
    assert len(result.value.segments) == 1


def test_invalid_output_is_repaired_once() -> None:
    fake = FakeCompletion('{"segments": [{"start": "00:01"}]}', VALID)
    result = LLMClient("gemini/x", completion_fn=fake).complete_structured(MESSAGES, RawTranscript)
    assert len(fake.calls) == 2
    repair = fake.calls[1]["messages"]
    assert repair[-2]["role"] == "assistant"
    assert "speaker" in repair[-1]["content"]
    assert result.cost_usd == pytest.approx(0.004)
    assert result.prompt_tokens == 20


def test_invalid_output_twice_raises() -> None:
    fake = FakeCompletion("not json", "still not json")
    with pytest.raises(LLMOutputError, match="gemini/x"):
        LLMClient("gemini/x", completion_fn=fake).complete_structured(MESSAGES, RawTranscript)


def test_transient_errors_are_retried() -> None:
    error = litellm.exceptions.RateLimitError("slow down", llm_provider="gemini", model="x")
    fake = FakeCompletion(error, VALID)
    result = LLMClient("gemini/x", completion_fn=fake).complete_structured(MESSAGES, RawTranscript)
    assert len(fake.calls) == 2
    assert result.value.segments


def test_non_transient_errors_are_not_retried() -> None:
    error = litellm.exceptions.AuthenticationError("bad key", llm_provider="gemini", model="x")
    fake = FakeCompletion(error, VALID)
    with pytest.raises(litellm.exceptions.AuthenticationError):
        LLMClient("gemini/x", completion_fn=fake).complete_structured(MESSAGES, RawTranscript)
    assert len(fake.calls) == 1


def test_unknown_cost_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_pricing(**_: Any) -> float:
        raise ValueError("model not mapped")

    monkeypatch.setattr(litellm, "completion_cost", no_pricing)
    fake = FakeCompletion(VALID)
    result = LLMClient("gemini/x", completion_fn=fake).complete_structured(MESSAGES, RawTranscript)
    assert result.cost_usd is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"a": 1}', '{"a": 1}'),
        ('```json\n{"a": 1}\n```', '{"a": 1}'),
        ('Kết quả: {"a": 1} xong', '{"a": 1}'),
        ("no json", "no json"),
    ],
)
def test_extract_json(raw: str, expected: str) -> None:
    assert extract_json(raw) == expected
