"""Optional meeting context (meeting.yaml) that improves transcription and minutes."""

from datetime import date as Date
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class ContextError(ValueError):
    pass


class Participant(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    role: str | None = None

    def label(self) -> str:
        return f"{self.name} ({self.role})" if self.role else self.name


class MeetingContext(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    title: str | None = None
    date: Date | None = None
    location: str | None = None
    participants: tuple[Participant, ...] = ()
    agenda: tuple[str, ...] = ()
    glossary: tuple[str, ...] = Field(default=(), description="Names/terms to spell correctly")
    notes: str | None = None

    @field_validator("participants", mode="before")
    @classmethod
    def _accept_plain_names(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [{"name": item} if isinstance(item, str) else item for item in value]
        return value


def load_context(path: Path) -> MeetingContext:
    if not path.is_file():
        raise ContextError(f"Context file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ContextError(f"Invalid YAML in {path}: {exc}") from exc
    if data is None:
        return MeetingContext()
    if not isinstance(data, dict):
        raise ContextError(f"{path} must contain a YAML mapping (key: value)")
    try:
        return MeetingContext.model_validate(data)
    except ValidationError as exc:
        raise ContextError(f"Invalid meeting context in {path}:\n{exc}") from exc
