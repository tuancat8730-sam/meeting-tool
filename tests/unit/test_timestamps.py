import pytest

from meeting_tool.utils.timestamps import (
    TimestampError,
    format_hms,
    format_mmss,
    format_srt,
    parse_clock,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("00:00", 0.0),
        ("01:05", 65.0),
        ("[12:34]", 754.0),
        ("1:02:03", 3723.0),
        ("75:30", 4530.0),
        ("00:05.5", 5.5),
        ("00:05,250", 5.25),
        (" 02:00 ", 120.0),
    ],
)
def test_parse_clock_valid(value: str, expected: float) -> None:
    assert parse_clock(value) == pytest.approx(expected)


@pytest.mark.parametrize("value", ["", "abc", "5", "00:60", "1:60:00", "-01:00", "00:5a"])
def test_parse_clock_invalid(value: str) -> None:
    with pytest.raises(TimestampError):
        parse_clock(value)


def test_format_hms() -> None:
    assert format_hms(0) == "00:00:00"
    assert format_hms(3723.9) == "01:02:03"


def test_format_mmss() -> None:
    assert format_mmss(65.4) == "01:05"
    assert format_mmss(1830) == "30:30"


def test_format_srt() -> None:
    assert format_srt(3723.4567) == "01:02:03,457"
    assert format_srt(59.9996) == "00:01:00,000"


@pytest.mark.parametrize("fn", [format_hms, format_mmss, format_srt])
def test_format_rejects_negative(fn: object) -> None:
    with pytest.raises(ValueError):
        fn(-1)  # type: ignore[operator]
