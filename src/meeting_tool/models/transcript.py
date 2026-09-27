from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class RawSegment(BaseModel):
    """Segment exactly as the LLM returns it: `start` is chunk-relative 'MM:SS'."""

    start: str
    speaker: str
    text: str
    nonverbal: str | None = None


class RawTranscript(BaseModel):
    """LLM response schema for one audio chunk."""

    segments: list[RawSegment]


class Segment(BaseModel):
    model_config = ConfigDict(frozen=True)

    start: float = Field(ge=0, description="Absolute seconds from the start of the recording")
    end: float | None = Field(default=None, ge=0)
    speaker: str
    text: str
    nonverbal: str | None = None


class ChunkTranscript(BaseModel):
    """Result for one chunk, persisted to the work dir so runs can resume."""

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=0)
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    segments: tuple[Segment, ...]
    model: str
    cost_usd: float | None
    timestamp_issues: int = 0


class Transcript(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_file: Path
    duration_s: float = Field(ge=0)
    model: str
    segments: tuple[Segment, ...]
    cost_usd: float | None = Field(description="None when pricing is unknown for any chunk")
    chunk_count: int = Field(ge=0)
    timestamp_issues: int = Field(ge=0)
    created_at: datetime

    @property
    def speakers(self) -> tuple[str, ...]:
        """Distinct speakers in order of first appearance."""
        return tuple(dict.fromkeys(s.speaker for s in self.segments))
