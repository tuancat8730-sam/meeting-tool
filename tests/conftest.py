from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
import stamina
from pydantic import BaseModel
from pydub.generators import Sine

from meeting_tool.llm.client import StructuredResult
from meeting_tool.models.transcript import RawSegment, RawTranscript


@pytest.fixture(autouse=True)
def _fast_retries() -> None:
    stamina.set_testing(True, attempts=3)


def raw(*segments: tuple[str, str, str]) -> RawTranscript:
    """raw(("00:05", "Anh Tuấn", "Xin chào"), ...) -> RawTranscript"""
    return RawTranscript(
        segments=[RawSegment(start=s, speaker=sp, text=t) for s, sp, t in segments]
    )


class FakeLLM:
    """Stands in for LLMClient. `responses` is a list consumed in order, or a callable."""

    def __init__(
        self,
        responses: Sequence[BaseModel] | Callable[[int, list[dict[str, Any]]], BaseModel],
        model: str = "fake/model",
        cost_usd: float | None = 0.01,
    ) -> None:
        self.model = model
        self._responses = responses
        self._cost = cost_usd
        self.calls: list[list[dict[str, Any]]] = []

    def complete_structured[T: BaseModel](
        self, messages: list[dict[str, Any]], schema: type[T]
    ) -> StructuredResult[T]:
        index = len(self.calls)
        self.calls.append(messages)
        if callable(self._responses):
            value = self._responses(index, messages)
        else:
            value = self._responses[index]
        assert isinstance(value, schema)
        return StructuredResult(
            value=value, cost_usd=self._cost, prompt_tokens=100, completion_tokens=50
        )


def user_text(messages: list[dict[str, Any]]) -> str:
    content = messages[-1]["content"]
    return "".join(part["text"] for part in content if part["type"] == "text")


@pytest.fixture
def sine_wav(tmp_path: Path) -> Callable[[float], Path]:
    def make(seconds: float) -> Path:
        path = tmp_path / f"tone_{seconds:g}s.wav"
        Sine(440).to_audio_segment(duration=int(seconds * 1000)).export(path, format="wav")
        return path

    return make
