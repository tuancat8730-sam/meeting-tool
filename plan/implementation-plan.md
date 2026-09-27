# Implementation Plan — Meeting Tool (audio → transcript → biên bản họp)

Based on [context-meeting-tool.md](../context-meeting-tool.md). Date: 2026-09-27.

## 0. Decisions (confirmed)

| Topic | Decision | Consequence |
|---|---|---|
| Language | **Vietnamese only** | All prompts written in Vietnamese; minutes output in Vietnamese; eval set = Vietnamese audio (multiple regional accents). |
| Privacy | **Cloud LLM APIs OK** | LiteLLM multimodal transcription (Gemini default). No local ASR in scope. |
| Interface | **CLI first, web later** | Phases 1–3 = CLI. Phase 4 = FastAPI + upload page, reusing the same core library. |
| Codebase | **Clean-room rewrite** | Reuse *ideas* from rivol/llm-transcribe (chunk + overlap, context carry-over, timestamp dedupe), not code. |

Assumed defaults (change if wrong):
- Typical meeting length ≤ 3h; must not break on longer files.
- Minutes follow a standard Vietnamese "biên bản họp" layout, rendered from a **Jinja2 template** so a company template can replace it without code changes.
- Default models: transcribe `gemini/gemini-2.5-flash`, minutes `gemini/gemini-2.5-pro` (both overridable).

## 1. Tech stack

- Python 3.13, **uv** (deps + venv), `pyproject.toml`, `ruff`, `mypy`, `pytest` + `pytest-cov`
- Typer + rich (CLI), **Pydantic v2** (`field_validator`, `model_validate_json`)
- pydub + FFmpeg (audio), LiteLLM (providers + `completion_cost` + `token_counter`), stamina (retry)
- Jinja2 (Markdown template), python-docx (Word export)
- Phase 4: FastAPI + a simple HTML upload page + background job runner

## 2. Project layout

```
meeting-tool/
├── pyproject.toml
├── src/meeting_tool/
│   ├── cli.py                  # Typer app: transcribe | minutes | run
│   ├── config.py               # Settings (env vars, model names, chunk sizes)
│   ├── models/
│   │   ├── audio.py            # AudioChunk
│   │   ├── transcript.py       # Segment, Transcript
│   │   ├── context.py          # MeetingContext (title, date, participants, agenda)
│   │   └── minutes.py          # Participant, Topic, Decision, ActionItem, MeetingMinutes
│   ├── audio/chunker.py        # load, validate format, split w/ overlap
│   ├── llm/
│   │   ├── client.py           # LiteLLM wrapper: retry, cost accounting, structured output
│   │   └── prompts/            # transcribe_vi.md, minutes_vi.md, minutes_reduce_vi.md
│   ├── transcribe/
│   │   ├── engine.py           # sequential chunk loop + context carry-over
│   │   └── merge.py            # relative→absolute timestamps, overlap dedupe, sanity checks
│   ├── minutes/
│   │   ├── generator.py        # single-pass or map-reduce
│   │   └── splitter.py         # split transcript by time windows / token budget
│   ├── export/
│   │   ├── transcript.py       # txt, srt, json
│   │   ├── markdown.py         # Jinja2 render
│   │   ├── docx.py
│   │   └── templates/bien_ban_hop.md.j2
│   └── utils/timestamps.py
└── tests/
    ├── unit/  integration/  e2e/
    └── fixtures/               # short synthetic audio, sample transcripts, recorded LLM responses
```

## 3. Key design choices

1. **Structured output for transcription too**, not only for minutes. Ask the model for JSON
   `{"segments": [{"start": "MM:SS", "speaker": str, "text": str, "nonverbal": str|null}]}` using a LiteLLM `response_format` Pydantic schema. This avoids the brittle regex parsing in the upstream repo. If a provider doesn't support schemas with audio, fall back to the `[MM:SS] Speaker: text` line format and a tolerant parser.
2. **Chunking**: 10 min chunks, 1 min overlap (both configurable). Export chunks as mono 16 kHz Opus/MP3 to keep the payload small.
3. **Context carry-over**: send the last ~60s of the previous chunk's transcript (converted to chunk-relative timestamps), plus the known speaker list, to keep speaker names consistent. Chunks run **sequentially** by default. A `--parallel` mode without carry-over, followed by a speaker-reconciliation LLM pass, comes later.
4. **Timestamp sanity**: clamp to the chunk's duration, enforce monotonic order, and dedupe the overlap by absolute timestamp plus text similarity (catches LLM timestamp drift).
5. **Speakers**: labels are "Người nói 1/2/…" unless names are inferred from the audio or from `MeetingContext.participants`. The minutes list the participants and flag inferred names.
6. **Minutes generation**: if the transcript's token count is ≤ 60% of the minutes model's context window, use a single pass. Otherwise use map-reduce: 30-min windows → partial minutes → reduce pass that merges and dedupes decisions and action items. Every decision and action item carries `timestamp_refs` back into the transcript.
7. **Meeting context input**: `--context meeting.yaml` (title, date, location, participants with roles, agenda). It is used by both the transcribe prompt (names, domain terms) and the minutes prompt.
8. **Cost tracking**: record per-call cost and tokens, split by stage (transcribe vs minutes). Show them in the CLI summary and write them to `*.report.json`.
9. **Intermediate artifacts are persisted** (per-chunk JSON). A crash at chunk 14 of 18 can resume with `--resume`, so earlier chunks aren't paid for twice.

## 4. Data model (Pydantic v2)

```python
class Segment(BaseModel):
    start: float  # absolute seconds
    end: float | None
    speaker: str
    text: str
    nonverbal: str | None = None


class Transcript(BaseModel):
    source_file: Path
    duration: float
    segments: list[Segment]
    model: str
    cost_usd: float


class Topic(BaseModel):
    heading: str
    summary: str
    timestamp_refs: list[str] = []


class Decision(BaseModel):
    summary: str
    timestamp_refs: list[str] = []


class ActionItem(BaseModel):
    description: str
    owner: str | None = None
    due_date: str | None = None  # keep as spoken ("thứ Sáu tuần sau")
    status: Literal["open", "done", "blocked"] = "open"
    timestamp_refs: list[str] = []


class MeetingMinutes(BaseModel):
    title: str
    meeting_date: date | None = None
    duration_seconds: float | None = None
    participants: list[Participant] = []
    summary: str
    topics: list[Topic] = []
    decisions: list[Decision] = []
    action_items: list[ActionItem] = []
    open_questions: list[str] = []
    model_used: str
    source_transcript_file: Path
```

The LLM fills only the content fields. Metadata such as model, source, and duration is set in code.

## 5. CLI

```bash
meeting-tool transcribe hop.m4a -c meeting.yaml -o out/ --export txt,srt,json
meeting-tool minutes out/hop.json -c meeting.yaml --format md,docx [--model ...]
meeting-tool run hop.m4a -c meeting.yaml -o out/          # full pipeline
# common: --model, --minutes-model, --chunk-minutes, --overlap-seconds, --resume, --dry-run (cost estimate)
```

## 6. Phases

### Phase 0 — Spike & evaluation (≈2–3 days) ⚠️ gate
- Collect 3–5 real Vietnamese recordings: 1 short (5 min), 1 with ≥4 speakers, 1 long (≥1h), different accents (Bắc/Trung/Nam).
- Write a throwaway script that sends one 10-min chunk to `gemini-2.5-flash`, `gemini-2.5-pro`, and `gpt-4o-audio-preview` using a draft Vietnamese prompt.
- Measure: word accuracy on a hand-corrected 5-min sample (target WER < 15%), speaker attribution, timestamp drift, cost per hour of audio, and latency.
- **Exit criteria**: a default model is picked and the quality is acceptable. If Vietnamese quality is poor, revisit (e.g. Whisper-large-v3 API for ASR + LLM diarization) before continuing.

### Phase 1 — Core transcript pipeline (≈1 week)
1. Scaffold: uv project, ruff/mypy/pytest config, `config.py` (env-based, validates API keys at startup).
2. `models/` + `utils/timestamps.py` — TDD, pure functions.
3. `audio/chunker.py` — format validation (m4a/mp3/wav/flac/ogg/aac), clear error if FFmpeg is missing, overlap math.
4. `llm/client.py` — LiteLLM wrapper with stamina retry (429/5xx/timeouts only), structured output, cost accounting.
5. `prompts/transcribe_vi.md` — Vietnamese rules: verbatim, keep filler words optional, speaker labels, non-verbal cues in Vietnamese, don't translate English terms.
6. `transcribe/engine.py` + `merge.py` — carry-over, dedupe, sanity checks, per-chunk persistence and `--resume`.
7. `export/transcript.py` — txt, **srt** (`HH:MM:SS,mmm`, end = next segment start, capped at 7s per cue), json.
8. `cli.py transcribe`.
- **Done when**: a 1h Vietnamese meeting produces txt/srt/json with no duplicated overlap lines, and the srt loads correctly in VLC.

### Phase 2 — Meeting minutes (≈1 week)
1. `models/minutes.py`, `models/context.py` (YAML loader + validation).
2. `prompts/minutes_vi.md` — extraction rules: only facts stated in the transcript, never invent owners or dates, cite timestamps, and use "Chưa xác định" for unknowns.
3. `minutes/generator.py` single-pass with schema-validated output (retry once on validation failure, feeding back the error).
4. `export/markdown.py` + `templates/bien_ban_hop.md.j2` (Thời gian · Địa điểm · Thành phần tham dự · Nội dung chính theo chủ đề · Kết luận/Quyết định · Phân công công việc (table) · Vấn đề còn mở).
5. `cli.py minutes` and `run`.
- **Done when**: minutes for the eval recordings are reviewed by a human; no hallucinated action items; every decision is traceable to a timestamp.

### Phase 3 — Hardening & UX (≈1 week)
1. Map-reduce for long transcripts (`splitter.py`, `minutes_reduce_vi.md`).
2. `.docx` export (python-docx, same sections as the template).
3. Cost report per stage and a `--dry-run` cost estimate before calling the API.
4. Optional `--parallel` transcription with a speaker-reconciliation pass.
5. Rich progress bars, friendly errors, README (vi).
- **Done when**: a 3h file completes end-to-end, coverage is ≥ 80%, and ruff/mypy are clean.

### Phase 4 — Web (later, separate plan)
FastAPI app: upload → background job (start with a simple in-process queue, move to RQ/Celery later) → job status → download txt/srt/md/docx. Include auth, upload size/type validation, file retention/deletion policy, and rate limiting. Optional Slack/email delivery.

## 7. Testing strategy

- **Unit**: timestamps, chunk math, merge/dedupe (including drift and duplicate cases), srt formatting, context YAML validation, template rendering, splitter.
- **Integration**: engine and generator with LiteLLM `mock_response` / recorded fixtures (no network in CI); CLI via Typer `CliRunner`.
- **E2E (manual/opt-in, `-m live`)**: real API on a 2-min Vietnamese fixture; skipped without an API key.
- **Quality eval** (not in CI): a script that reruns the Phase 0 set and diffs minutes against human-approved references, to catch prompt regressions.

## 8. Risks

| Risk | Mitigation |
|---|---|
| Vietnamese ASR quality / accents | Phase 0 gate; domain terms and names passed via context; model switchable. |
| LLM timestamp drift/hallucination | Clamp + monotonic enforcement + text-similarity dedupe. |
| Wrong speaker attribution | Participant list in context; carry-over; flag uncertainty in minutes. |
| Hallucinated decisions/owners | Strict prompt + required `timestamp_refs` + human review before sending. |
| Cost on long meetings | Compressed audio, `--dry-run` estimate, `--resume`, cheap transcribe model. |
| Confidential audio sent to cloud | Documented in README; use paid-tier APIs (no training on data); keep no audio on disk after the run unless asked. |
| Provider payload limits | Chunk size configurable; Gemini File API upload for large chunks if needed. |

## 9. What I need from you

1. 3–5 Vietnamese sample recordings (plus consent to send them to Gemini/OpenAI) for Phase 0.
2. API keys available as env vars: `GEMINI_API_KEY` (and `OPENAI_API_KEY` if we compare models).
3. Your company's biên bản template, if one exists. Otherwise the default template is used.
