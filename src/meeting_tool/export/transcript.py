"""Transcript exporters: plain text, SubRip subtitles, JSON."""

from collections.abc import Callable, Sequence
from pathlib import Path

from meeting_tool.models.transcript import Segment, Transcript
from meeting_tool.utils.timestamps import format_hms, format_srt

# Subtitle cues end at the next segment, but no later than the time needed to read them
READING_CHARS_PER_SECOND = 15
MIN_CUE_SECONDS = 1.5


def _line_text(segment: Segment) -> str:
    suffix = f" ({segment.nonverbal})" if segment.nonverbal else ""
    return f"{segment.speaker}: {segment.text}{suffix}"


def to_txt(transcript: Transcript) -> str:
    lines = [f"[{format_hms(s.start)}] {_line_text(s)}" for s in transcript.segments]
    return "\n".join(lines) + "\n"


def _cue_end(segment: Segment, fallback_end: float) -> float:
    reading = max(MIN_CUE_SECONDS, len(segment.text) / READING_CHARS_PER_SECOND)
    limit = segment.end if segment.end is not None else fallback_end
    return max(min(limit, segment.start + reading), segment.start + 0.5)


def to_srt(transcript: Transcript) -> str:
    cues = []
    for number, segment in enumerate(transcript.segments, start=1):
        end = _cue_end(segment, transcript.duration_s)
        cues.append(
            f"{number}\n{format_srt(segment.start)} --> {format_srt(end)}\n{_line_text(segment)}\n"
        )
    return "\n".join(cues)


def to_json(transcript: Transcript) -> str:
    return transcript.model_dump_json(indent=2) + "\n"


EXPORTERS: dict[str, Callable[[Transcript], str]] = {"txt": to_txt, "srt": to_srt, "json": to_json}


def write_exports(
    transcript: Transcript, out_dir: Path, stem: str, formats: Sequence[str]
) -> list[Path]:
    unknown = [f for f in formats if f not in EXPORTERS]
    if unknown:
        raise ValueError(
            f"Unknown export format(s): {', '.join(unknown)}. Use: {', '.join(EXPORTERS)}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        path.write_text(EXPORTERS[fmt](transcript), encoding="utf-8")
        paths.append(path)
    return paths
