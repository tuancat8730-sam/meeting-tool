"""Turn per-chunk LLM output into one clean, absolute-time transcript."""

import itertools
import re
import unicodedata
from collections.abc import Sequence
from difflib import SequenceMatcher

from meeting_tool.models.transcript import ChunkTranscript, RawTranscript, Segment
from meeting_tool.utils.timestamps import TimestampError, parse_clock

UNKNOWN_SPEAKER = "Không xác định"
DUPLICATE_SIMILARITY = 0.85
# How many segments on each side of a chunk boundary are checked for duplicates
BOUNDARY_WINDOW = 3
# A duplicate must start within this many seconds of the kept segment's time span
MAX_DRIFT_SECONDS = 10.0


def _with_ends(segments: Sequence[Segment], final_end: float) -> tuple[Segment, ...]:
    """Set each segment's end to the next segment's start (last one ends at `final_end`)."""
    ends = [s.start for s in segments[1:]] + [final_end]
    return tuple(
        s.model_copy(update={"end": max(end, s.start)})
        for s, end in zip(segments, ends, strict=True)
    )


def to_absolute(
    raw: RawTranscript, chunk_start: float, chunk_end: float
) -> tuple[tuple[Segment, ...], int]:
    """Convert chunk-relative LLM output to absolute segments. Returns (segments, issue count).

    Malformed, out-of-range and backwards timestamps are repaired (never dropped, so no text is
    lost) and counted as issues.
    """
    duration = chunk_end - chunk_start
    segments: list[Segment] = []
    issues = 0
    previous = 0.0
    for item in raw.segments:
        text = item.text.strip()
        if not text:
            continue
        try:
            relative = parse_clock(item.start)
        except TimestampError:
            issues += 1
            relative = previous
        if relative > duration:
            issues += 1
            relative = duration
        if relative < previous:
            issues += 1
            relative = previous
        previous = relative
        segments.append(
            Segment(
                start=chunk_start + relative,
                speaker=item.speaker.strip() or UNKNOWN_SPEAKER,
                text=text,
                nonverbal=(item.nonverbal or "").strip() or None,
            )
        )
    return _with_ends(segments, chunk_end), issues


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).casefold()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text)).strip()


def _overlaps_in_time(candidate: Segment, kept: Segment) -> bool:
    kept_end = kept.end if kept.end is not None else kept.start
    return kept.start - MAX_DRIFT_SECONDS <= candidate.start <= kept_end + MAX_DRIFT_SECONDS


def _is_duplicate(candidate: Segment, kept: Sequence[Segment]) -> bool:
    """Same (or contained) text at roughly the same time → the overlap was transcribed twice."""
    new = _normalize(candidate.text)
    if not new:
        return True
    for segment in kept:
        if not _overlaps_in_time(candidate, segment):
            continue
        old = _normalize(segment.text)
        if new in old or SequenceMatcher(None, new, old).ratio() >= DUPLICATE_SIMILARITY:
            return True
    return False


def _completion_of(cut: Segment, candidates: Sequence[Segment]) -> Segment | None:
    """Find a longer version of `cut` (a sentence truncated by the end of its chunk's audio)."""
    old = _normalize(cut.text)
    for candidate in candidates:
        new = _normalize(candidate.text)
        if len(new) <= len(old) or not _overlaps_in_time(candidate, cut):
            continue
        if SequenceMatcher(None, old, new[: len(old)]).ratio() >= DUPLICATE_SIMILARITY:
            return candidate
    return None


def merge_chunks(chunks: Sequence[ChunkTranscript]) -> tuple[Segment, ...]:
    """Merge overlapping chunks: cut each overlap at its midpoint, then drop near-duplicates
    (timestamp drift or re-split sentences) around the cut.

    The previous chunk's final segment may be a sentence cut off where its audio stops; if the
    next chunk heard the whole sentence, that complete version replaces it.
    """
    if not chunks:
        return ()
    merged: list[Segment] = list(_with_ends(chunks[0].segments, chunks[0].end))
    for previous, current in itertools.pairwise(chunks):
        boundary = (current.start + previous.end) / 2
        merged = [s for s in merged if s.start < boundary]
        current_segments = _with_ends(current.segments, current.end)
        incoming = [s for s in current_segments if s.start >= boundary]
        if merged and merged[-1].end == previous.end:
            completion = _completion_of(merged[-1], current_segments)
            if completion is not None:
                merged[-1] = completion
                incoming = [s for s in incoming if s is not completion]
        recent = merged[-BOUNDARY_WINDOW:]
        head = [s for s in incoming[:BOUNDARY_WINDOW] if not _is_duplicate(s, recent)]
        merged += head + incoming[BOUNDARY_WINDOW:]
    return _with_ends(merged, chunks[-1].end)
