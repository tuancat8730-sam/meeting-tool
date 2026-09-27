from dataclasses import dataclass, field


@dataclass(frozen=True)
class AudioChunk:
    """One encoded slice of the source audio. `start`/`end` are absolute seconds."""

    index: int
    start: float
    end: float
    data: bytes = field(repr=False)
    format: str = "mp3"

    @property
    def duration(self) -> float:
        return self.end - self.start
