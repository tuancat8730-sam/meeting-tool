"""Builds the system prompt and per-chunk messages for transcription."""

import base64
from collections.abc import Sequence
from importlib import resources
from typing import Any

from meeting_tool.models.audio import AudioChunk
from meeting_tool.models.context import MeetingContext
from meeting_tool.models.transcript import Segment
from meeting_tool.utils.timestamps import format_hms, format_mmss

PROMPT_FILE = "transcribe_vi.md"


def load_prompt(name: str) -> str:
    return (resources.files("meeting_tool") / "llm" / "prompts" / name).read_text("utf-8")


def _bullets(items: Sequence[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def build_system_prompt(context: MeetingContext | None) -> str:
    parts = [load_prompt(PROMPT_FILE).rstrip()]
    if context is None:
        return parts[0] + "\n"
    if context.title:
        parts.append(f"## Cuộc họp\n\n{context.title}")
    if context.participants:
        labels = [p.label() for p in context.participants]
        parts.append(f"## Người tham dự dự kiến\n\n{_bullets(labels)}")
    if context.agenda:
        parts.append(f"## Chương trình họp\n\n{_bullets(context.agenda)}")
    if context.glossary:
        parts.append(f"## Thuật ngữ, tên riêng\n\n{_bullets(context.glossary)}")
    if context.notes:
        parts.append(f"## Ghi chú\n\n{context.notes}")
    return "\n\n".join(parts) + "\n"


def _user_text(
    chunk: AudioChunk,
    total_chunks: int,
    previous_tail: Sequence[Segment],
    known_speakers: Sequence[str],
) -> str:
    lines = [
        f"Đây là phần {chunk.index + 1}/{total_chunks} của cuộc họp, bắt đầu tại "
        f"{format_hms(chunk.start)} và dài {format_mmss(chunk.duration)}. "
        "Thời gian `start` tính từ đầu đoạn âm thanh này (00:00)."
    ]
    if previous_tail:
        tail_lines = [
            f"[{format_mmss(max(0.0, s.start - chunk.start))}] {s.speaker}: {s.text}"
            for s in previous_tail
        ]
        lines += [
            "",
            "## Ngữ cảnh từ phần trước",
            "Phần đầu của đoạn âm thanh này trùng với cuối phần trước và đã được chép như sau "
            "(thời gian đã quy đổi theo đoạn hiện tại):",
            *tail_lines,
            "Hãy dùng đúng các tên người nói này cho cùng giọng nói. "
            "Vẫn chép lời toàn bộ đoạn âm thanh, kể cả phần trùng lặp.",
        ]
    if known_speakers:
        lines += ["", f"Người nói đã xuất hiện trước đó: {', '.join(known_speakers)}."]
    return "\n".join(lines)


def build_messages(
    system_prompt: str,
    chunk: AudioChunk,
    total_chunks: int,
    previous_tail: Sequence[Segment],
    known_speakers: Sequence[str],
) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": _user_text(chunk, total_chunks, previous_tail, known_speakers),
                },
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": base64.b64encode(chunk.data).decode("ascii"),
                        "format": chunk.format,
                    },
                },
            ],
        },
    ]
