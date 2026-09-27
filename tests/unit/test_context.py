from datetime import date
from pathlib import Path

import pytest

from meeting_tool.models.context import ContextError, MeetingContext, Participant, load_context


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "meeting.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_full_context(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
title: Họp sprint 42
date: 2026-09-27
location: Phòng 3A
participants:
  - Anh Tuấn (PM)
  - {name: Chị Lan, role: QA}
agenda: [Demo, Kế hoạch sprint]
glossary: [Kubernetes, VNPay]
""",
    )
    ctx = load_context(path)
    assert ctx.title == "Họp sprint 42"
    assert ctx.date == date(2026, 9, 27)
    assert ctx.participants == (
        Participant(name="Anh Tuấn (PM)"),
        Participant(name="Chị Lan", role="QA"),
    )
    assert ctx.agenda == ("Demo", "Kế hoạch sprint")
    assert ctx.glossary == ("Kubernetes", "VNPay")


def test_empty_file_gives_empty_context(tmp_path: Path) -> None:
    assert load_context(write(tmp_path, "")) == MeetingContext()


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ContextError, match="not found"):
        load_context(tmp_path / "nope.yaml")


def test_invalid_yaml(tmp_path: Path) -> None:
    with pytest.raises(ContextError, match="YAML"):
        load_context(write(tmp_path, "title: [unclosed"))


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ContextError, match="participant"):
        load_context(write(tmp_path, "participant: [A]"))


def test_top_level_must_be_mapping(tmp_path: Path) -> None:
    with pytest.raises(ContextError, match="mapping"):
        load_context(write(tmp_path, "- a\n- b\n"))
