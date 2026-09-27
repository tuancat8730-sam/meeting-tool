"""Phase 0 spike: compare multimodal LLMs on Vietnamese meeting transcription.

Throwaway code — not part of the product. Sends one audio window to each model and reports
latency, cost, cost per audio-hour, timestamp sanity, speaker count and (optionally) WER.

Usage:
    uv run python spikes/phase0_eval.py samples/hop1.m4a \
        --models gemini/gemini-2.5-flash,gemini/gemini-2.5-pro \
        --start-minute 0 --minutes 10 \
        --reference samples/hop1_ref.txt \
        --participants "Anh Tuấn (PM), Chị Lan (QA)"
"""

import base64
import io
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Annotated, Any

import litellm
import typer
from pydantic import BaseModel, ValidationError
from pydub import AudioSegment
from rich.console import Console
from rich.table import Table

from meeting_tool.config import MissingApiKeyError, require_api_key

SPIKE_DIR = Path(__file__).parent
PROMPT_PATH = SPIKE_DIR / "transcribe_prompt_vi.md"
RESULTS_DIR = SPIKE_DIR / "results"
SUPPORTED_SUFFIXES = {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".aac"}

console = Console()


class SpikeSegment(BaseModel):
    start: str
    speaker: str
    text: str
    nonverbal: str | None = None


class SpikeTranscript(BaseModel):
    segments: list[SpikeSegment]


def parse_timestamp(value: str) -> float | None:
    """'MM:SS' or 'HH:MM:SS' -> seconds; None if malformed."""
    parts = value.strip().strip("[]").split(":")
    if len(parts) not in (2, 3) or not all(p.isdigit() for p in parts):
        return None
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + int(part)
    return seconds


def normalize_for_wer(text: str) -> str:
    """Lowercase, NFC, drop punctuation and bracketed tags; keeps Vietnamese diacritics."""
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_window(path: Path, start_minute: float, minutes: float) -> tuple[bytes, float]:
    """Slice the window, downmix to mono 16 kHz MP3. Returns (mp3 bytes, window seconds)."""
    audio = AudioSegment.from_file(path)
    start_ms = int(start_minute * 60_000)
    if start_ms >= len(audio):
        raise typer.BadParameter(f"--start-minute is past the end of {path.name}")
    window = audio[start_ms : start_ms + int(minutes * 60_000)]
    window = window.set_channels(1).set_frame_rate(16_000)
    buf = io.BytesIO()
    window.export(buf, format="mp3", bitrate="48k")
    return buf.getvalue(), len(window) / 1000


def build_messages(audio_mp3: bytes, participants: str | None) -> list[dict[str, Any]]:
    system = PROMPT_PATH.read_text(encoding="utf-8")
    if participants:
        system += f"\n\n## Người tham dự dự kiến\n\n{participants}\n"
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Chép lời đoạn âm thanh cuộc họp sau."},
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": base64.b64encode(audio_mp3).decode(),
                        "format": "mp3",
                    },
                },
            ],
        },
    ]


def extract_json(raw: str) -> str:
    """Strip ```json fences some models add despite instructions."""
    match = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    return match.group(1) if match else raw


def timestamp_issues(segments: list[SpikeSegment], window_seconds: float) -> dict[str, int]:
    malformed = out_of_range = non_monotonic = 0
    previous = -1.0
    for seg in segments:
        ts = parse_timestamp(seg.start)
        if ts is None:
            malformed += 1
            continue
        if ts > window_seconds + 1:
            out_of_range += 1
        if ts < previous:
            non_monotonic += 1
        previous = ts
    return {"malformed": malformed, "out_of_range": out_of_range, "non_monotonic": non_monotonic}


def compute_wer(reference_path: Path | None, segments: list[SpikeSegment]) -> float | None:
    if reference_path is None:
        return None
    import jiwer

    reference = normalize_for_wer(reference_path.read_text(encoding="utf-8"))
    hypothesis = normalize_for_wer(" ".join(s.text for s in segments))
    if not reference:
        raise typer.BadParameter(f"Reference file {reference_path} is empty")
    return float(jiwer.wer(reference, hypothesis))


def run_model(model: str, messages: list[dict[str, Any]]) -> tuple[str, float, float]:
    """Returns (raw content, latency seconds, cost usd)."""
    response_format: Any = (
        SpikeTranscript
        if litellm.supports_response_schema(model=model)
        else {"type": "json_object"}
    )
    started = time.perf_counter()
    response = litellm.completion(
        model=model,
        messages=messages,
        response_format=response_format,
        temperature=0,
        timeout=600,
        num_retries=2,
    )
    latency = time.perf_counter() - started
    try:
        cost = float(litellm.completion_cost(completion_response=response))
    except Exception as exc:  # pricing may be missing for new models
        console.print(f"[yellow]No cost data for {model}: {exc}[/yellow]")
        cost = float("nan")
    return response.choices[0].message.content or "", latency, cost


def evaluate(
    model: str,
    audio_path: Path,
    messages: list[dict[str, Any]],
    window_seconds: float,
    reference: Path | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"model": model, "audio": audio_path.name, "window_s": window_seconds}
    try:
        require_api_key(model)
        raw, latency, cost = run_model(model, messages)
    except (MissingApiKeyError, litellm.exceptions.APIError, litellm.exceptions.OpenAIError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    result |= {
        "latency_s": round(latency, 1),
        "cost_usd": cost,
        "cost_per_audio_hour_usd": cost * 3600 / window_seconds,
    }
    try:
        transcript = SpikeTranscript.model_validate_json(extract_json(raw))
    except ValidationError as exc:
        result |= {"error": f"Invalid JSON output: {exc.error_count()} errors", "raw": raw}
        return result

    result |= {
        "segments": [s.model_dump() for s in transcript.segments],
        "speakers": sorted({s.speaker for s in transcript.segments}),
        "timestamp_issues": timestamp_issues(transcript.segments, window_seconds),
        "wer": compute_wer(reference, transcript.segments),
    }
    return result


def save_result(result: dict[str, Any]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{result['model'].replace('/', '_')}__{Path(result['audio']).stem}"
    json_path = RESULTS_DIR / f"{stem}.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if "segments" in result:
        lines = [f"[{s['start']}] {s['speaker']}: {s['text']}" for s in result["segments"]]
        (RESULTS_DIR / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path


def print_summary(results: list[dict[str, Any]]) -> None:
    table = Table(title="Phase 0 — transcription comparison")
    for col in (
        "model",
        "latency",
        "cost",
        "$/audio-h",
        "segments",
        "speakers",
        "ts issues",
        "WER",
    ):
        table.add_column(col)
    for r in results:
        if "error" in r:
            table.add_row(r["model"], f"[red]{r['error'][:60]}[/red]", *[""] * 6)
            continue
        issues = sum(r["timestamp_issues"].values())
        table.add_row(
            r["model"],
            f"{r['latency_s']}s",
            f"${r['cost_usd']:.4f}",
            f"${r['cost_per_audio_hour_usd']:.3f}",
            str(len(r["segments"])),
            str(len(r["speakers"])),
            str(issues),
            "-" if r["wer"] is None else f"{r['wer']:.1%}",
        )
    console.print(table)


def main(
    audio: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    models: Annotated[str, typer.Option(help="Comma-separated LiteLLM model names")] = (
        "gemini/gemini-2.5-flash,gemini/gemini-2.5-pro"
    ),
    start_minute: Annotated[float, typer.Option(min=0)] = 0.0,
    minutes: Annotated[float, typer.Option(min=0.5, max=30)] = 10.0,
    reference: Annotated[
        Path | None, typer.Option(exists=True, dir_okay=False, help="Hand-corrected transcript")
    ] = None,
    participants: Annotated[str | None, typer.Option(help="Expected attendees")] = None,
) -> None:
    if audio.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise typer.BadParameter(f"Unsupported format {audio.suffix}; use {SUPPORTED_SUFFIXES}")

    audio_mp3, window_seconds = load_window(audio, start_minute, minutes)
    console.print(f"Window: {window_seconds:.0f}s, payload {len(audio_mp3) / 1e6:.1f} MB (mp3)")
    messages = build_messages(audio_mp3, participants)

    results = []
    for model in (m.strip() for m in models.split(",") if m.strip()):
        console.print(f"→ {model} ...")
        result = evaluate(model, audio, messages, window_seconds, reference)
        console.print(f"  saved {save_result(result)}")
        results.append(result)
    print_summary(results)


if __name__ == "__main__":
    typer.run(main)
