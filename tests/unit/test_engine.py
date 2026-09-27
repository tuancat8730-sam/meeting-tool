from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from meeting_tool.config import Settings
from meeting_tool.models.transcript import RawTranscript
from meeting_tool.transcribe.engine import ResumeMismatchError, TranscriptionEngine
from tests.conftest import FakeLLM, raw, user_text

# 25s audio, 15s chunks, 5s overlap → chunks (0-15) and (10-25); boundary at 12.5s
SETTINGS = Settings(_env_file=None, chunk_minutes=0.25, overlap_seconds=5)
RESPONSES = [
    raw(("00:01", "Anh Tuấn", "Bắt đầu họp"), ("00:11", "Chị Lan", "Em báo cáo")),
    raw(("00:01", "Chị Lan", "Em báo cáo"), ("00:08", "Anh Tuấn", "Cảm ơn em")),
]


@pytest.fixture
def audio(sine_wav: Callable[[float], Path]) -> Path:
    return sine_wav(25)


def test_transcribe_two_chunks(audio: Path, tmp_path: Path) -> None:
    llm = FakeLLM(RESPONSES)
    progress: list[tuple[int, int]] = []
    transcript = TranscriptionEngine(llm, SETTINGS).transcribe(
        audio, work_dir=tmp_path / "work", on_progress=lambda d, t: progress.append((d, t))
    )
    assert [(s.start, s.speaker, s.text) for s in transcript.segments] == [
        (1, "Anh Tuấn", "Bắt đầu họp"),
        (11, "Chị Lan", "Em báo cáo"),
        (18, "Anh Tuấn", "Cảm ơn em"),
    ]
    assert transcript.duration_s == 25
    assert transcript.chunk_count == 2
    assert transcript.cost_usd == pytest.approx(0.02)
    assert transcript.speakers == ("Anh Tuấn", "Chị Lan")
    assert progress == [(1, 2), (2, 2)]


def test_second_chunk_gets_carry_over(audio: Path, tmp_path: Path) -> None:
    llm = FakeLLM(RESPONSES)
    TranscriptionEngine(llm, SETTINGS).transcribe(audio, work_dir=tmp_path / "work")
    second = user_text(llm.calls[1])
    assert "[00:01] Chị Lan: Em báo cáo" in second  # 11s abs - 10s chunk start
    assert "Bắt đầu họp" not in second  # outside the overlap window
    assert "Anh Tuấn, Chị Lan" in second


def test_chunk_results_are_persisted_and_resumable(audio: Path, tmp_path: Path) -> None:
    work = tmp_path / "work"
    first = TranscriptionEngine(FakeLLM(RESPONSES), SETTINGS).transcribe(audio, work_dir=work)
    assert sorted(p.name for p in work.iterdir()) == [
        "chunk_000.json",
        "chunk_001.json",
        "manifest.json",
    ]

    def must_not_call(index: int, messages: list[dict[str, Any]]) -> RawTranscript:
        raise AssertionError("resume should not call the LLM")

    resumed = TranscriptionEngine(FakeLLM(must_not_call), SETTINGS).transcribe(
        audio, work_dir=work, resume=True
    )
    assert resumed.segments == first.segments


def test_resume_only_calls_missing_chunks(audio: Path, tmp_path: Path) -> None:
    work = tmp_path / "work"
    TranscriptionEngine(FakeLLM(RESPONSES), SETTINGS).transcribe(audio, work_dir=work)
    (work / "chunk_001.json").unlink()
    llm = FakeLLM([RESPONSES[1]])
    TranscriptionEngine(llm, SETTINGS).transcribe(audio, work_dir=work, resume=True)
    assert len(llm.calls) == 1
    assert "[00:01] Chị Lan: Em báo cáo" in user_text(llm.calls[0])


def test_resume_with_different_settings_fails(audio: Path, tmp_path: Path) -> None:
    work = tmp_path / "work"
    TranscriptionEngine(FakeLLM(RESPONSES), SETTINGS).transcribe(audio, work_dir=work)
    changed = Settings(_env_file=None, chunk_minutes=0.25, overlap_seconds=4)
    with pytest.raises(ResumeMismatchError, match="overlap_seconds"):
        TranscriptionEngine(FakeLLM(RESPONSES), changed).transcribe(
            audio, work_dir=work, resume=True
        )


def test_fresh_run_clears_stale_chunks(audio: Path, tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    (work / "chunk_009.json").write_text("{}")
    TranscriptionEngine(FakeLLM(RESPONSES), SETTINGS).transcribe(audio, work_dir=work)
    assert not (work / "chunk_009.json").exists()


def test_unknown_cost_is_reported_as_none(audio: Path, tmp_path: Path) -> None:
    llm = FakeLLM(RESPONSES, cost_usd=None)
    transcript = TranscriptionEngine(llm, SETTINGS).transcribe(audio, work_dir=tmp_path / "w")
    assert transcript.cost_usd is None
