import base64

from meeting_tool.models.audio import AudioChunk
from meeting_tool.models.context import MeetingContext, Participant
from meeting_tool.models.transcript import Segment
from meeting_tool.transcribe.prompt import build_messages, build_system_prompt

CHUNK = AudioChunk(index=1, start=540.0, end=1140.0, data=b"mp3bytes")


def test_system_prompt_without_context() -> None:
    prompt = build_system_prompt(None)
    assert "tiếng Việt" in prompt
    assert "Người tham dự" not in prompt


def test_system_prompt_with_context() -> None:
    ctx = MeetingContext(
        title="Họp sprint",
        participants=(Participant(name="Anh Tuấn", role="PM"), Participant(name="Chị Lan")),
        agenda=("Demo",),
        glossary=("VNPay",),
    )
    prompt = build_system_prompt(ctx)
    assert "Anh Tuấn (PM)" in prompt
    assert "Chị Lan" in prompt
    assert "VNPay" in prompt
    assert "Demo" in prompt
    assert "Họp sprint" in prompt


def test_first_chunk_has_no_carry_over() -> None:
    first = AudioChunk(index=0, start=0.0, end=600.0, data=b"x")
    messages = build_messages("SYS", first, total_chunks=3, previous_tail=(), known_speakers=())
    text = messages[1]["content"][0]["text"]
    assert "1/3" in text
    assert "phần trước" not in text


def test_carry_over_uses_chunk_relative_times() -> None:
    tail = (Segment(start=560.0, speaker="Anh Tuấn", text="Chốt thứ Sáu"),)
    messages = build_messages(
        "SYS", CHUNK, total_chunks=3, previous_tail=tail, known_speakers=("Anh Tuấn", "Chị Lan")
    )
    assert messages[0] == {"role": "system", "content": "SYS"}
    text = messages[1]["content"][0]["text"]
    assert "2/3" in text
    assert "00:09:00" in text  # absolute start of chunk
    assert "[00:20] Anh Tuấn: Chốt thứ Sáu" in text  # 560 - 540 = 20s
    assert "Anh Tuấn, Chị Lan" in text


def test_audio_part_is_base64_mp3() -> None:
    messages = build_messages("SYS", CHUNK, total_chunks=1, previous_tail=(), known_speakers=())
    audio = messages[1]["content"][1]
    assert audio["type"] == "input_audio"
    assert audio["input_audio"]["format"] == "mp3"
    assert base64.b64decode(audio["input_audio"]["data"]) == b"mp3bytes"
