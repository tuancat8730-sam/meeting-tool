import io
import itertools
from collections.abc import Callable
from pathlib import Path

import pytest
from pydub import AudioSegment

from meeting_tool.audio.chunker import AudioLoadError, encode_chunk, load_audio, plan_chunks


@pytest.mark.parametrize(
    ("duration", "expected"),
    [
        (300, [(0, 300)]),
        (600, [(0, 600)]),
        (1200, [(0, 600), (540, 1140), (1080, 1200)]),
        (603, [(0, 603)]),  # 3s tail is folded into the previous chunk
        (1143, [(0, 600), (540, 1143)]),
    ],
)
def test_plan_chunks(duration: float, expected: list[tuple[float, float]]) -> None:
    assert plan_chunks(duration, chunk_s=600, overlap_s=60) == expected


def test_plan_chunks_covers_whole_duration_with_overlap() -> None:
    spans = plan_chunks(10_000, chunk_s=600, overlap_s=60)
    assert spans[0][0] == 0
    assert spans[-1][1] == 10_000
    for (_, prev_end), (next_start, _) in itertools.pairwise(spans):
        assert prev_end - next_start == pytest.approx(60)


@pytest.mark.parametrize(
    ("duration", "chunk_s", "overlap_s"),
    [(0, 600, 60), (100, 60, 60), (100, 60, -1), (100, 0, 0)],
)
def test_plan_chunks_rejects_invalid(duration: float, chunk_s: float, overlap_s: float) -> None:
    with pytest.raises(ValueError):
        plan_chunks(duration, chunk_s=chunk_s, overlap_s=overlap_s)


def test_load_audio_ok(sine_wav: Callable[[float], Path]) -> None:
    audio = load_audio(sine_wav(2))
    assert len(audio) == 2000


def test_load_audio_missing(tmp_path: Path) -> None:
    with pytest.raises(AudioLoadError, match="not found"):
        load_audio(tmp_path / "x.mp3")


def test_load_audio_unsupported_suffix(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("hi")
    with pytest.raises(AudioLoadError, match="Unsupported"):
        load_audio(path)


def test_load_audio_corrupt(tmp_path: Path) -> None:
    path = tmp_path / "broken.mp3"
    path.write_bytes(b"not really audio")
    with pytest.raises(AudioLoadError, match="decode"):
        load_audio(path)


def test_load_audio_without_ffmpeg(
    monkeypatch: pytest.MonkeyPatch, sine_wav: Callable[[float], Path]
) -> None:
    monkeypatch.setattr("meeting_tool.audio.chunker.shutil.which", lambda _: None)
    with pytest.raises(AudioLoadError, match="FFmpeg"):
        load_audio(sine_wav(1))


def test_encode_chunk_is_mono_16k_mp3(sine_wav: Callable[[float], Path]) -> None:
    audio = load_audio(sine_wav(5))
    chunk = encode_chunk(audio, index=1, start=1.0, end=4.0)
    assert (chunk.index, chunk.start, chunk.end, chunk.format) == (1, 1.0, 4.0, "mp3")
    assert chunk.duration == 3.0
    decoded = AudioSegment.from_file(io.BytesIO(chunk.data), format="mp3")
    assert decoded.channels == 1
    assert decoded.frame_rate == 16_000
    assert abs(len(decoded) - 3000) < 100
