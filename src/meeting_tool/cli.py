"""CLI entry point: `meeting-tool transcribe` (Phase 1); `minutes` and `run` come in Phase 2."""

from pathlib import Path
from typing import Annotated

import litellm
import typer
from pydantic import ValidationError
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from meeting_tool.audio.chunker import AudioLoadError
from meeting_tool.config import MissingApiKeyError, Settings, require_api_key
from meeting_tool.export.transcript import EXPORTERS, write_exports
from meeting_tool.llm.client import LLMClient, LLMOutputError
from meeting_tool.models.context import ContextError, load_context
from meeting_tool.models.transcript import Transcript
from meeting_tool.transcribe.engine import ResumeMismatchError, StructuredLLM, TranscriptionEngine
from meeting_tool.utils.timestamps import format_hms

app = typer.Typer(
    help="Vietnamese meeting audio -> transcript -> biên bản họp.", no_args_is_help=True
)
console = Console()


class UsageError(ValueError):
    pass


# Errors with messages written for the user; anything else is a bug and keeps its traceback
USER_ERRORS: tuple[type[Exception], ...] = (
    UsageError,
    AudioLoadError,
    ContextError,
    LLMOutputError,
    MissingApiKeyError,
    ResumeMismatchError,
    litellm.exceptions.AuthenticationError,
    litellm.exceptions.BadRequestError,
    litellm.exceptions.RateLimitError,
    litellm.exceptions.APIConnectionError,
)


def make_client(model: str, timeout: float) -> StructuredLLM:
    return LLMClient(model, timeout_s=timeout)


def _parse_formats(value: str) -> tuple[str, ...]:
    formats = tuple(dict.fromkeys(f.strip().lower() for f in value.split(",") if f.strip()))
    unknown = [f for f in formats if f not in EXPORTERS]
    if not formats or unknown:
        raise UsageError(
            f"Invalid export format(s): {', '.join(unknown) or '(none)'}. "
            f"Choose from: {', '.join(EXPORTERS)}"
        )
    return formats


def _build_settings(**overrides: object) -> Settings:
    try:
        return Settings(**{k: v for k, v in overrides.items() if v is not None})  # type: ignore[arg-type]
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(map(str, e['loc'])) or 'settings'}: {e['msg']}" for e in exc.errors()
        )
        raise UsageError(f"Invalid settings: {details}") from exc


def _print_summary(transcript: Transcript, paths: list[Path]) -> None:
    cost = "unknown" if transcript.cost_usd is None else f"${transcript.cost_usd:.4f}"
    console.print(
        f"[green]Done.[/green] {format_hms(transcript.duration_s)} audio, "
        f"{len(transcript.segments)} segments, {len(transcript.speakers)} speakers, "
        f"{transcript.chunk_count} chunks, cost {cost}"
    )
    if transcript.timestamp_issues:
        console.print(
            f"[yellow]{transcript.timestamp_issues} timestamp(s) from the model were "
            "repaired; check the transcript around chunk boundaries.[/yellow]"
        )
    for path in paths:
        console.print(f"  → {path}")


@app.command()
def transcribe(
    audio: Annotated[
        Path, typer.Argument(help="Meeting recording (m4a, mp3, wav, flac, ogg, aac)")
    ],
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path("out"),
    context: Annotated[
        Path | None, typer.Option("--context", "-c", help="meeting.yaml with participants etc.")
    ] = None,
    export: Annotated[str, typer.Option("--export", "-e", help="Comma-separated: txt,srt,json")] = (
        "txt,srt,json"
    ),
    model: Annotated[str | None, typer.Option("--model", "-m", help="LiteLLM model name")] = None,
    chunk_minutes: Annotated[float | None, typer.Option(help="Chunk length (minutes)")] = None,
    overlap_seconds: Annotated[float | None, typer.Option(help="Chunk overlap (seconds)")] = None,
    resume: Annotated[
        bool, typer.Option(help="Reuse finished chunks from a previous interrupted run")
    ] = False,
) -> None:
    """Transcribe a meeting recording into a timestamped, speaker-labelled transcript."""
    try:
        formats = _parse_formats(export)
        settings = _build_settings(
            transcribe_model=model, chunk_minutes=chunk_minutes, overlap_seconds=overlap_seconds
        )
        require_api_key(settings.transcribe_model)
        meeting = load_context(context) if context else None
        engine = TranscriptionEngine(
            make_client(settings.transcribe_model, settings.llm_timeout_seconds), settings, meeting
        )
        console.print(f"Transcribing [bold]{audio.name}[/bold] with {settings.transcribe_model}")
        with Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("chunks", total=None)
            transcript = engine.transcribe(
                audio,
                work_dir=output_dir / f"{audio.stem}.chunks",
                resume=resume,
                on_progress=lambda done, total: progress.update(task, completed=done, total=total),
            )
        paths = write_exports(transcript, output_dir, audio.stem, formats)
    except USER_ERRORS as exc:
        console.print(f"[red]Error:[/red] {exc}")
        if isinstance(exc, LLMOutputError | litellm.exceptions.RateLimitError):
            console.print("Finished chunks are saved; rerun with --resume to continue.")
        raise typer.Exit(1) from exc
    _print_summary(transcript, paths)


@app.command()
def version() -> None:
    """Print the installed version."""
    from importlib.metadata import version as pkg_version

    typer.echo(pkg_version("meeting-tool"))
