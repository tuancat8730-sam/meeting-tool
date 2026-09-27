# meeting-tool

Turns a Vietnamese meeting recording into a timestamped, speaker-labelled transcript
(and, from Phase 2, a biên bản họp) using multimodal LLMs via LiteLLM.

Status: **Phase 1 — transcription** (see [plan/implementation-plan.md](plan/implementation-plan.md)).

## Setup

Requires [uv](https://docs.astral.sh/uv/) and FFmpeg (`sudo apt install ffmpeg`).

```bash
uv sync
export GEMINI_API_KEY=...        # or OPENAI_API_KEY / ANTHROPIC_API_KEY for other models
```

## Transcribe

```bash
uv run meeting-tool transcribe hop.m4a -c examples/meeting.yaml -o out/
```

Writes `out/hop.txt`, `out/hop.srt` and `out/hop.json`.

| Option | Default | |
|---|---|---|
| `-c, --context` | none | `meeting.yaml` with title, participants, agenda, glossary ([example](examples/meeting.yaml)) — improves speaker names and spelling |
| `-e, --export` | `txt,srt,json` | Output formats |
| `-m, --model` | `gemini/gemini-2.5-flash` | Any LiteLLM model that accepts audio |
| `--chunk-minutes` | `10` | Audio is sent in chunks of this length |
| `--overlap-seconds` | `60` | Overlap between chunks, used to keep speakers consistent and avoid cut sentences |
| `--resume` | off | Reuse chunks finished by an interrupted run (same file, model and chunk settings) |

Defaults can also be set with environment variables: `MEETING_TOOL_TRANSCRIBE_MODEL`,
`MEETING_TOOL_CHUNK_MINUTES`, `MEETING_TOOL_OVERLAP_SECONDS`, `MEETING_TOOL_LLM_TIMEOUT_SECONDS`.

Per-chunk results are kept in `out/<name>.chunks/`, so a failed run can continue with `--resume`
without paying again for finished chunks.

## Privacy

Audio is sent to the chosen LLM provider. Use a paid API tier whose terms exclude training on
your data, and don't commit recordings — `.gitignore` excludes audio files and `out/`.

## Development

```bash
uv run pytest --cov=meeting_tool      # unit + integration tests (no network)
uv run ruff check . && uv run ruff format --check .
uv run mypy src spikes tests
```

Phase 0 model comparison: `uv run python spikes/phase0_eval.py --help`.
