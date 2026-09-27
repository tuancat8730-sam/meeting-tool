from datetime import UTC, datetime
from pathlib import Path

import pytest

from meeting_tool.export.transcript import to_srt, to_txt, write_exports
from meeting_tool.models.transcript import Segment, Transcript


def make_transcript() -> Transcript:
    return Transcript(
        source_file=Path("hop.m4a"),
        duration_s=3700.0,
        model="gemini/gemini-2.5-flash",
        segments=(
            Segment(start=5.0, end=6.0, speaker="Anh Tuấn", text="Bắt đầu nhé."),
            Segment(start=6.0, end=120.0, speaker="Chị Lan", text="Vâng.", nonverbal="cười"),
            Segment(start=3661.5, end=3700.0, speaker="Anh Tuấn", text="Kết thúc."),
        ),
        cost_usd=0.12,
        chunk_count=7,
        timestamp_issues=0,
        created_at=datetime(2026, 9, 27, tzinfo=UTC),
    )


def test_txt() -> None:
    assert to_txt(make_transcript()).splitlines() == [
        "[00:00:05] Anh Tuấn: Bắt đầu nhé.",
        "[00:00:06] Chị Lan: Vâng. (cười)",
        "[01:01:01] Anh Tuấn: Kết thúc.",
    ]


def test_srt_cues() -> None:
    cues = to_srt(make_transcript()).strip().split("\n\n")
    assert cues[0].splitlines() == [
        "1",
        "00:00:05,000 --> 00:00:06,000",
        "Anh Tuấn: Bắt đầu nhé.",
    ]
    # short text does not linger until the next segment 2 minutes later
    assert cues[1].splitlines()[1] == "00:00:06,000 --> 00:00:07,500"
    assert cues[1].splitlines()[2] == "Chị Lan: Vâng. (cười)"
    assert cues[2].splitlines()[0] == "3"


def test_srt_long_text_gets_reading_time() -> None:
    text = "x" * 150  # 150 chars at 15 chars/s = 10s
    transcript = make_transcript().model_copy(
        update={"segments": (Segment(start=0, end=60, speaker="A", text=text),)}
    )
    assert "00:00:00,000 --> 00:00:10,000" in to_srt(transcript)


def test_json_round_trip(tmp_path: Path) -> None:
    transcript = make_transcript()
    paths = write_exports(transcript, tmp_path, "hop", ("json",))
    assert paths == [tmp_path / "hop.json"]
    assert Transcript.model_validate_json(paths[0].read_text(encoding="utf-8")) == transcript


def test_write_all_formats(tmp_path: Path) -> None:
    paths = write_exports(make_transcript(), tmp_path / "out", "hop", ("txt", "srt", "json"))
    assert [p.name for p in paths] == ["hop.txt", "hop.srt", "hop.json"]
    assert all(p.stat().st_size > 0 for p in paths)


def test_unknown_format_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="docx"):
        write_exports(make_transcript(), tmp_path, "hop", ("docx",))
