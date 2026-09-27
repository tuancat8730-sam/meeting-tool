from collections.abc import Callable
from pathlib import Path

import pytest
from typer.testing import CliRunner

from meeting_tool import cli
from tests.conftest import FakeLLM, raw

runner = CliRunner()
FAST = ["--chunk-minutes", "0.25", "--overlap-seconds", "5"]


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    llm = FakeLLM(lambda i, _: raw((f"00:0{i + 1}", "Anh Tuấn", f"Câu {i}")))
    monkeypatch.setattr(cli, "make_client", lambda model, timeout: llm)
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    return llm


def test_transcribe_writes_all_exports(
    fake_llm: FakeLLM, sine_wav: Callable[[float], Path], tmp_path: Path
) -> None:
    audio = sine_wav(25)
    out = tmp_path / "out"
    result = runner.invoke(cli.app, ["transcribe", str(audio), "-o", str(out), *FAST])
    assert result.exit_code == 0, result.output
    stem = audio.stem
    assert {p.name for p in out.iterdir()} >= {f"{stem}.txt", f"{stem}.srt", f"{stem}.json"}
    assert "Câu 0" in (out / f"{stem}.txt").read_text(encoding="utf-8")
    assert len(fake_llm.calls) == 2


def test_transcribe_with_context(
    fake_llm: FakeLLM, sine_wav: Callable[[float], Path], tmp_path: Path
) -> None:
    ctx = tmp_path / "meeting.yaml"
    ctx.write_text("participants: [Anh Tuấn (PM)]\n", encoding="utf-8")
    args = ["transcribe", str(sine_wav(5)), "-o", str(tmp_path), "-c", str(ctx), "-e", "txt"]
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert "Anh Tuấn (PM)" in fake_llm.calls[0][0]["content"]


def test_missing_api_key(
    monkeypatch: pytest.MonkeyPatch, sine_wav: Callable[[float], Path], tmp_path: Path
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    result = runner.invoke(cli.app, ["transcribe", str(sine_wav(2)), "-o", str(tmp_path)])
    assert result.exit_code == 1
    assert "GEMINI_API_KEY" in result.output


def test_bad_export_format(
    fake_llm: FakeLLM, sine_wav: Callable[[float], Path], tmp_path: Path
) -> None:
    result = runner.invoke(cli.app, ["transcribe", str(sine_wav(2)), "-e", "txt,pdf"])
    assert result.exit_code == 1
    assert "pdf" in result.output
    assert fake_llm.calls == []


def test_bad_settings(fake_llm: FakeLLM, sine_wav: Callable[[float], Path]) -> None:
    args = ["transcribe", str(sine_wav(2)), "--chunk-minutes", "1", "--overlap-seconds", "90"]
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 1
    assert "overlap_seconds" in result.output


def test_bad_audio(fake_llm: FakeLLM, tmp_path: Path) -> None:
    broken = tmp_path / "broken.mp3"
    broken.write_bytes(b"junk")
    result = runner.invoke(cli.app, ["transcribe", str(broken), "-o", str(tmp_path)])
    assert result.exit_code == 1
    assert "decode" in result.output
