import itertools

import pytest

from meeting_tool.models.transcript import ChunkTranscript, Segment
from meeting_tool.transcribe.merge import UNKNOWN_SPEAKER, merge_chunks, to_absolute
from tests.conftest import raw


def seg(start: float, text: str, speaker: str = "A") -> Segment:
    return Segment(start=start, speaker=speaker, text=text)


def chunk(index: int, start: float, end: float, *segments: Segment) -> ChunkTranscript:
    return ChunkTranscript(
        index=index, start=start, end=end, segments=segments, model="m", cost_usd=0.0
    )


class TestToAbsolute:
    def test_offsets_and_ends(self) -> None:
        segments, issues = to_absolute(
            raw(("00:01", "A", "một"), ("00:30", "B", "hai")), chunk_start=540, chunk_end=1140
        )
        assert [(s.start, s.end, s.speaker) for s in segments] == [
            (541, 570, "A"),
            (570, 1140, "B"),
        ]
        assert issues == 0

    def test_malformed_timestamp_reuses_previous(self) -> None:
        segments, issues = to_absolute(
            raw(("00:10", "A", "một"), ("??", "A", "hai")), chunk_start=0, chunk_end=60
        )
        assert [s.start for s in segments] == [10, 10]
        assert issues == 1

    def test_out_of_range_is_clamped(self) -> None:
        segments, issues = to_absolute(raw(("09:00", "A", "x")), chunk_start=100, chunk_end=160)
        assert segments[0].start == 160
        assert issues == 1

    def test_non_monotonic_is_clamped(self) -> None:
        segments, issues = to_absolute(
            raw(("00:20", "A", "một"), ("00:05", "B", "hai")), chunk_start=0, chunk_end=60
        )
        assert [s.start for s in segments] == [20, 20]
        assert issues == 1

    def test_empty_text_dropped_and_blank_speaker_labelled(self) -> None:
        segments, _ = to_absolute(
            raw(("00:01", "A", "  "), ("00:02", " ", " có nội dung ")), chunk_start=0, chunk_end=10
        )
        assert len(segments) == 1
        assert segments[0].speaker == UNKNOWN_SPEAKER
        assert segments[0].text == "có nội dung"


class TestMergeChunks:
    def test_single_chunk_passthrough(self) -> None:
        merged = merge_chunks([chunk(0, 0, 100, seg(0, "a"), seg(50, "b"))])
        assert [(s.text, s.end) for s in merged] == [("a", 50), ("b", 100)]

    def test_overlap_split_at_midpoint(self) -> None:
        # chunk0 covers 0-600, chunk1 covers 540-1140 → boundary at 570
        c0 = chunk(0, 0, 600, seg(10, "mở đầu"), seg(560, "câu A"), seg(580, "câu B cũ"))
        c1 = chunk(1, 540, 1140, seg(560, "câu A"), seg(575, "câu B mới"), seg(700, "câu C"))
        merged = merge_chunks([c0, c1])
        assert [s.text for s in merged] == ["mở đầu", "câu A", "câu B mới", "câu C"]
        assert merged[-1].end == 1140

    def test_drifted_duplicate_is_dropped(self) -> None:
        c0 = chunk(0, 0, 600, seg(565, "Chúng ta chốt deploy vào thứ Sáu nhé"))
        c1 = chunk(1, 540, 1140, seg(571, "Chúng ta chốt deploy vào thứ sáu nhé."), seg(600, "Ok"))
        merged = merge_chunks([c0, c1])
        assert [s.text for s in merged] == ["Chúng ta chốt deploy vào thứ Sáu nhé", "Ok"]

    def test_fragment_of_previous_segment_is_dropped(self) -> None:
        long = "Về ngân sách thì quý này mình còn khoảng hai trăm triệu cho hạ tầng"
        c0 = chunk(0, 0, 600, seg(560, long))
        c1 = chunk(1, 540, 1140, seg(572, "còn khoảng hai trăm triệu cho hạ tầng"), seg(590, "Ừ"))
        assert [s.text for s in merge_chunks([c0, c1])] == [long, "Ừ"]

    @pytest.mark.parametrize("complete_start", [19, 21])  # before / after the 20s boundary
    def test_sentence_cut_by_chunk_end_is_replaced(self, complete_start: float) -> None:
        # chunk0's audio stops at 24s, mid-sentence; chunk1 (16-37s) heard all of it
        cut = "Chị Lan kiểm thử luồng thanh toán, bao gồm cả trường hợp"
        full = "Chị Lan kiểm thử luồng thanh toán, bao gồm cả trường hợp hoàn tiền."
        c0 = chunk(0, 0, 24, seg(0, "Mở đầu"), seg(19, cut))
        c1 = chunk(1, 16, 37, seg(16, "hợp lệ"), seg(complete_start, full), seg(25, "Về ngân sách"))
        merged = merge_chunks([c0, c1])
        assert [s.text for s in merged] == ["Mở đầu", full, "Về ngân sách"]

    def test_last_segment_kept_when_next_chunk_differs(self) -> None:
        c0 = chunk(0, 0, 24, seg(19, "Chị Lan kiểm thử"))
        c1 = chunk(1, 16, 37, seg(19, "Một câu khác hẳn"), seg(25, "Tiếp"))
        assert [s.text for s in merge_chunks([c0, c1])] == ["Chị Lan kiểm thử", "Tiếp"]

    def test_empty_input(self) -> None:
        assert merge_chunks([]) == ()

    @pytest.mark.parametrize("count", [3, 5])
    def test_many_chunks_keep_order(self, count: int) -> None:
        chunks = [
            chunk(i, i * 540, i * 540 + 600, seg(i * 540 + 100, f"chunk {i}")) for i in range(count)
        ]
        merged = merge_chunks(chunks)
        assert [s.text for s in merged] == [f"chunk {i}" for i in range(count)]
        assert all(a.start <= b.start for a, b in itertools.pairwise(merged))
