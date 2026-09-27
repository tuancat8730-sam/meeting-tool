"""Orchestrates chunked transcription with context carry-over and resumable progress."""

import hashlib
import logging
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, ValidationError

from meeting_tool.audio.chunker import encode_chunk, load_audio, plan_chunks
from meeting_tool.config import Settings
from meeting_tool.llm.client import StructuredResult
from meeting_tool.models.context import MeetingContext
from meeting_tool.models.transcript import ChunkTranscript, RawTranscript, Segment, Transcript
from meeting_tool.transcribe.merge import merge_chunks, to_absolute
from meeting_tool.transcribe.prompt import build_messages, build_system_prompt

logger = logging.getLogger(__name__)

MANIFEST_FILE = "manifest.json"
ProgressFn = Callable[[int, int], None]


class StructuredLLM(Protocol):
    model: str

    def complete_structured[T: BaseModel](
        self, messages: list[dict[str, Any]], schema: type[T]
    ) -> StructuredResult[T]: ...


class ResumeMismatchError(RuntimeError):
    pass


class RunManifest(BaseModel):
    """Identifies a run; resuming is only allowed when this matches exactly."""

    model_config = ConfigDict(frozen=True)

    source_sha256: str
    source_size: int
    model: str
    chunk_seconds: float
    overlap_seconds: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_atomic(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def _prepare_work_dir(work_dir: Path, manifest: RunManifest, resume: bool) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = work_dir / MANIFEST_FILE
    if resume and manifest_path.exists():
        try:
            saved = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        except ValidationError as exc:
            raise ResumeMismatchError(f"Corrupt {manifest_path}; rerun without --resume") from exc
        if saved != manifest:
            changed = [
                name
                for name in RunManifest.model_fields
                if getattr(saved, name) != getattr(manifest, name)
            ]
            raise ResumeMismatchError(
                f"Cannot resume: {', '.join(changed)} changed since the previous run. "
                "Rerun without --resume to start over."
            )
    elif not resume:
        for stale in work_dir.glob("chunk_*.json"):
            stale.unlink()
    _write_atomic(manifest_path, manifest.model_dump_json(indent=2))


def _load_chunk(path: Path) -> ChunkTranscript | None:
    try:
        return ChunkTranscript.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError):
        logger.warning("Ignoring unreadable chunk file %s", path)
        return None


def _tail(chunk: ChunkTranscript, overlap_s: float) -> tuple[Segment, ...]:
    return tuple(s for s in chunk.segments if s.start >= chunk.end - overlap_s)


def _total_cost(chunks: Sequence[ChunkTranscript]) -> float | None:
    costs = [c.cost_usd for c in chunks]
    return None if any(c is None for c in costs) else sum(c for c in costs if c is not None)


class TranscriptionEngine:
    def __init__(
        self, llm: StructuredLLM, settings: Settings, context: MeetingContext | None = None
    ) -> None:
        self._llm = llm
        self._settings = settings
        self._system_prompt = build_system_prompt(context)

    def transcribe(
        self,
        audio_path: Path,
        work_dir: Path,
        resume: bool = False,
        on_progress: ProgressFn | None = None,
    ) -> Transcript:
        chunk_s = self._settings.chunk_minutes * 60
        overlap_s = self._settings.overlap_seconds
        audio = load_audio(audio_path)
        duration = len(audio) / 1000
        spans = plan_chunks(duration, chunk_s=chunk_s, overlap_s=overlap_s)
        manifest = RunManifest(
            source_sha256=_sha256(audio_path),
            source_size=audio_path.stat().st_size,
            model=self._llm.model,
            chunk_seconds=chunk_s,
            overlap_seconds=overlap_s,
        )
        _prepare_work_dir(work_dir, manifest, resume)

        done: list[ChunkTranscript] = []
        speakers: dict[str, None] = {}
        tail: tuple[Segment, ...] = ()
        for index, (start, end) in enumerate(spans):
            path = work_dir / f"chunk_{index:03d}.json"
            result = _load_chunk(path) if resume and path.exists() else None
            if result is None:
                result = self._transcribe_chunk(
                    audio, index, (start, end), len(spans), tail, speakers
                )
                _write_atomic(path, result.model_dump_json(indent=2))
            done.append(result)
            speakers.update(dict.fromkeys(s.speaker for s in result.segments))
            tail = _tail(result, overlap_s)
            if on_progress:
                on_progress(index + 1, len(spans))

        return Transcript(
            source_file=audio_path,
            duration_s=duration,
            model=self._llm.model,
            segments=merge_chunks(done),
            cost_usd=_total_cost(done),
            chunk_count=len(done),
            timestamp_issues=sum(c.timestamp_issues for c in done),
            created_at=datetime.now(UTC),
        )

    def _transcribe_chunk(
        self,
        audio: Any,
        index: int,
        span: tuple[float, float],
        total: int,
        tail: tuple[Segment, ...],
        speakers: dict[str, None],
    ) -> ChunkTranscript:
        start, end = span
        chunk = encode_chunk(audio, index, start, end)
        messages = build_messages(self._system_prompt, chunk, total, tail, tuple(speakers))
        response = self._llm.complete_structured(messages, RawTranscript)
        segments, issues = to_absolute(response.value, start, end)
        return ChunkTranscript(
            index=index,
            start=start,
            end=end,
            segments=segments,
            model=self._llm.model,
            cost_usd=response.cost_usd,
            timestamp_issues=issues,
        )
