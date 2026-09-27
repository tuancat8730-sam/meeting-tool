"""Timestamp parsing and formatting. Internally all times are float seconds."""

import re

_CLOCK_RE = re.compile(r"^\[?\s*(?:(\d+):)?(\d{1,3}):(\d{1,2})(?:[.,](\d{1,3}))?\s*\]?$")


class TimestampError(ValueError):
    pass


def parse_clock(value: str) -> float:
    """Parse 'MM:SS', 'HH:MM:SS', optionally bracketed or with '.mmm'/',mmm' fraction."""
    match = _CLOCK_RE.match(value.strip())
    if match is None:
        raise TimestampError(f"Invalid timestamp: {value!r}")
    hours, minutes, seconds, fraction = match.groups()
    if int(seconds) >= 60 or (hours is not None and int(minutes) >= 60):
        raise TimestampError(f"Invalid timestamp: {value!r}")
    total: float = int(hours or 0) * 3600 + int(minutes) * 60 + int(seconds)
    if fraction:
        total += int(fraction.ljust(3, "0")) / 1000
    return float(total)


def _require_non_negative(seconds: float) -> None:
    if seconds < 0:
        raise ValueError(f"Timestamp must be non-negative, got {seconds}")


def format_hms(seconds: float) -> str:
    """3723.9 -> '01:02:03' (truncates fractions)."""
    _require_non_negative(seconds)
    whole = int(seconds)
    return f"{whole // 3600:02d}:{whole % 3600 // 60:02d}:{whole % 60:02d}"


def format_mmss(seconds: float) -> str:
    """65.4 -> '01:05'. Minutes are not wrapped into hours (for chunk-relative times)."""
    _require_non_negative(seconds)
    whole = int(seconds)
    return f"{whole // 60:02d}:{whole % 60:02d}"


def format_srt(seconds: float) -> str:
    """3723.4567 -> '01:02:03,457'."""
    _require_non_negative(seconds)
    total_ms = round(seconds * 1000)
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, ms = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"
