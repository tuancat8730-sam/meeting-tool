"""Load audio and split it into overlapping, compressed chunks for the LLM."""

import io
import shutil
from pathlib import Path

from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

from meeting_tool.models.audio import AudioChunk

SUPPORTED_SUFFIXES = frozenset({".m4a", ".mp3", ".wav", ".flac", ".ogg", ".aac"})
# A final chunk adding less new audio than this is folded into the previous chunk
MIN_TAIL_SECONDS = 5.0
# Speech-friendly encoding: small payload, no audible loss for transcription
SAMPLE_RATE = 16_000
BITRATE = "48k"


class AudioLoadError(RuntimeError):
    pass


def plan_chunks(duration: float, chunk_s: float, overlap_s: float) -> list[tuple[float, float]]:
    """Return (start, end) spans in seconds; consecutive spans overlap by `overlap_s`."""
    if duration <= 0:
        raise ValueError("duration must be positive")
    if chunk_s <= 0 or not 0 <= overlap_s < chunk_s:
        raise ValueError("need chunk_s > 0 and 0 <= overlap_s < chunk_s")

    spans: list[tuple[float, float]] = []
    start = 0.0
    while True:
        end = min(start + chunk_s, duration)
        if duration - end < MIN_TAIL_SECONDS:
            spans.append((start, duration))
            return spans
        spans.append((start, end))
        start += chunk_s - overlap_s


def load_audio(path: Path) -> AudioSegment:
    if not path.is_file():
        raise AudioLoadError(f"Audio file not found: {path}")
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise AudioLoadError(f"Unsupported audio format '{path.suffix}'. Supported: {supported}")
    if shutil.which("ffmpeg") is None:
        raise AudioLoadError(
            "FFmpeg not found on PATH. Install it, e.g. `sudo apt install ffmpeg`."
        )
    try:
        audio = AudioSegment.from_file(path)
    except (CouldntDecodeError, IndexError, OSError) as exc:
        raise AudioLoadError(f"Could not decode audio file {path}: {exc}") from exc
    if len(audio) == 0:
        raise AudioLoadError(f"Audio file is empty: {path}")
    return audio


def encode_chunk(audio: AudioSegment, index: int, start: float, end: float) -> AudioChunk:
    """Slice [start, end) seconds, downmix to mono 16 kHz and encode as MP3."""
    window = audio[int(start * 1000) : int(end * 1000)]
    window = window.set_channels(1).set_frame_rate(SAMPLE_RATE)
    buffer = io.BytesIO()
    window.export(buffer, format="mp3", bitrate=BITRATE)
    return AudioChunk(index=index, start=start, end=end, data=buffer.getvalue())
