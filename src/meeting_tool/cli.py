"""CLI entry point. Commands are added in Phase 1 (transcribe) and Phase 2 (minutes, run)."""

import typer

app = typer.Typer(
    help="Vietnamese meeting audio -> transcript -> biên bản họp.", no_args_is_help=True
)


@app.command()
def version() -> None:
    """Print the installed version."""
    from importlib.metadata import version as pkg_version

    typer.echo(pkg_version("meeting-tool"))
