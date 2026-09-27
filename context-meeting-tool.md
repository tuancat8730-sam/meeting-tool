# Context: Ứng dụng chuyển ghi âm cuộc họp → Transcript → Meeting Minotes tự động

> Dự án: **llm-transcribe** (Cowork Project)
> Tài liệu này phân tích codebase tham khảo [rivol/llm-transcribe](https://github.com/rivol/llm-transcribe) và đề xuất kiến trúc/roadmap để mở rộng thành ứng dụng hoàn chỉnh: **audio cuộc họp → transcript có timestamp/speaker → biên bản họp (meeting minutes) tự động**.
> Cập nhật lần đầu: 2026-09-27

---

## 1. Mục tiêu sản phẩm

Người dùng có file ghi âm cuộc họp (m4a, mp3, wav, ...). Ứng dụng cần:

1. **Transcribe**: chuyển audio → text có timestamp, phân biệt người nói (speaker), giữ lại ngữ cảnh cảm xúc/non-verbal nếu có ích.
2. **Summarize**: từ transcript, tự động sinh **biên bản họp (meeting minutes)** có cấu trúc: người tham dự, tóm tắt nội dung, các quyết định, action items (kèm người phụ trách/hạn), câu hỏi còn mở.
3. Xuất ra định dạng dễ dùng: `.txt`/`.srt` cho transcript, `.md`/`.docx` cho biên bản họp.

Đây là bài toán 2 giai đoạn (2-stage pipeline): **ASR-by-LLM** (đã có sẵn phần lớn trong llm-transcribe) rồi đến **LLM summarization** (chưa có trong repo gốc — cần xây thêm).

---

## 2. Phân tích repo tham khảo: rivol/llm-transcribe

### 2.1 Tổng quan

- Python CLI tool (Typer), version `0.1.0`, Python ≥ 3.13.
- Không dùng ASR truyền thống (Whisper, v.v.) mà gửi **audio trực tiếp cho LLM đa phương thức** (Gemini/OpenAI/Anthropic qua LiteLLM) và yêu cầu model tự transcribe kèm timestamp + speaker + non-verbal cues (laughs, sighs, tone...).
- License: repo hiện không có file `LICENSE` — cần xác nhận điều khoản sử dụng lại code trước khi build sản phẩm dựa trên nó.

### 2.2 Kiến trúc & luồng dữ liệu

```
Audio File → AudioProcessor (pydub) → ChunkData[] (10 phút/chunk, overlap 1 phút)
                ↓
        TranscriptionEngine (transcriber.py) — orchestration
                ↓  (mỗi chunk)
        LLMClient (LiteLLM) → gửi audio (base64) + system prompt + context chunk trước
                ↓
        TranscriptionResult (Pydantic) → parse "[MM:SS] Speaker: text (non-verbal)"
                ↓
        OutputHandler → gộp, khử trùng lặp overlap → ghi .txt / .json / .report.txt
```

### 2.3 Các module chính (đáng tái sử dụng)

| Module | Vai trò | Ghi chú tái sử dụng |
|---|---|---|
| `models.py` | Pydantic models: `Config`, `ChunkData`, `TranscriptionLine`, `TranscriptionResult`, `TranscriptionJob` | Dùng lại gần như nguyên vẹn; sẽ cần thêm `MeetingMinutes` model (mục 4). |
| `audio.py` | Chia audio thành chunk 10 phút, overlap 1 phút bằng `pydub`, giữ audio in-memory | Tái sử dụng trực tiếp. Cần FFmpeg cài sẵn. |
| `llm_client.py` | Wrapper LiteLLM: encode audio base64, system prompt rất chi tiết (13 quy tắc transcribe + emotion/non-verbal), retry qua `stamina`, cost tracking | Tái sử dụng cho bước 1 (transcription). Prompt engineering ở đây là phần giá trị nhất của repo. |
| `transcriber.py` | `TranscriptionEngine`: điều phối tạo job, xử lý từng chunk tuần tự, trích "1 phút cuối" của chunk trước làm context cho chunk sau để giữ nhất quán tên speaker | Tái sử dụng; là nơi sẽ nối thêm bước sinh meeting minutes sau khi `process_job` hoàn tất. |
| `output.py` | Format kết quả, khử trùng lặp theo timestamp overlap, export `txt/json/report` | Cần mở rộng thêm export `srt` và `meeting_minutes.md/.docx`. |
| `cli.py` | Typer CLI: `-m model`, `-o output`, `-c context`, `--export txt,json,report`, `--test` | Cần thêm flag mới, ví dụ `--minutes` hoặc lệnh con `minutes`. |
| `timestamp_utils.py` | Format/parse `[HH:MM:SS]`, chuyển đổi tương đối ↔ tuyệt đối | Tái sử dụng. |

### 2.4 Cách xử lý ngữ cảnh giữa các chunk (quan trọng)

- Chunk dài 10 phút, overlap 1 phút → bước nhảy (step) thực tế 9 phút/chunk.
- Sau khi transcribe chunk N, lấy **1 phút cuối** của kết quả làm "context" gửi kèm chunk N+1, yêu cầu LLM lặp lại y nguyên phần context rồi transcribe tiếp — giúp giữ tên speaker nhất quán và tránh mất mạch hội thoại. Đổi timestamp tuyệt đối → tương đối trước khi đưa vào prompt (vì mỗi chunk audio gửi cho LLM đều "bắt đầu từ 00:00").
- Khử trùng lặp cuối cùng dựa trên so sánh timestamp tuyệt đối tăng dần (`output.py::deduplicate_overlapping_content`).

### 2.5 Định dạng output hiện tại

```
[01:23:45] John: We should start the meeting now.
[01:23:52] Sarah: I agree. Let me pull up the agenda.
```
Có thể kèm non-verbal: `(laughs)`, `(sighs, concerned)`, ...

### 2.6 Model/Provider hỗ trợ

Qua LiteLLM: Gemini (`gemini-2.5-flash`, `gemini-1.5-pro` — mặc định `gemini-2.5-flash`), OpenAI (`gpt-4o`, `gpt-4o-mini`), Anthropic (`claude-3-5-sonnet`, `claude-3-5-haiku`), và 100+ provider khác. API key qua biến môi trường (`GOOGLE_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).

### 2.7 Điểm mạnh

- Không cần deploy/host mô hình ASR riêng; tận dụng LLM đa phương thức có sẵn → triển khai nhanh, chất lượng transcript tốt với ngữ cảnh + cảm xúc mà Whisper thuần không có.
- Kiến trúc module hoá rõ ràng, dễ mở rộng (đã tách audio/LLM/orchestration/output).
- Có xử lý retry, cost tracking, test suite khá đầy đủ (audio chunking, context extraction, timestamp, output formatting).
- Tự động xử lý file dài (nhiều giờ) nhờ chunking + sliding window.

### 2.8 Hạn chế / rủi ro cần lưu ý khi mở rộng

- **Chưa có bước tóm tắt/biên bản họp** — chỉ dừng ở transcript thô. Đây chính là phần cần xây thêm.
- **Chưa có speaker diarization thực sự** (không tách theo giọng nói bằng audio embedding) — speaker label do LLM "đoán" qua ngữ cảnh hội thoại (tên được nhắc tới, giọng điệu câu chữ), có thể nhầm lẫn khi nhiều người nói giọng giống nhau hoặc không tự giới thiệu tên.
- **Chi phí & giới hạn kích thước request**: toàn bộ audio mỗi chunk (~10 phút, base64) được nhồi vào 1 request LLM → với cuộc họp nhiều giờ sẽ tốn kém, và cần theo dõi giới hạn payload/token của từng provider.
- **Xử lý tuần tự (sequential)** từng chunk — không có xử lý song song/async, cuộc họp dài sẽ chậm (tài liệu ghi "Future: async LLM calls if needed" nhưng chưa làm).
- **Không hỗ trợ tiếng Việt rõ ràng** trong prompt/test — cần kiểm thử thực tế với audio tiếng Việt (cả giọng vùng miền), vì prompt & ví dụ trong `llm_client.py` toàn bằng tiếng Anh.
- **Không có output `.srt`** dù mô tả dự án hiện tại của bạn có nhắc tới srt — cần viết thêm formatter riêng (timestamp `HH:MM:SS,mmm` + index dòng).
- **Bảo mật/riêng tư**: audio được gửi thẳng (base64) tới API bên thứ 3 (Google/OpenAI/Anthropic) — cần cân nhắc chính sách dữ liệu nếu cuộc họp có nội dung nhạy cảm/nội bộ.
- **Không có UI** — chỉ CLI. Nếu người dùng cuối không phải dev, cần thêm giao diện (web app, hoặc tích hợp bot Zoom/Meet/Teams).
- Dùng Pydantic v1-style `@validator` (deprecated ở Pydantic v2) — nên cân nhắc migrate sang `field_validator` nếu phát triển tiếp lâu dài.

---

## 3. Yêu cầu cho ứng dụng mới (mở rộng)

### 3.1 Input
- File audio: m4a, mp3, wav (đã hỗ trợ), có thể thêm flac/ogg/aac (pydub đã hỗ trợ, chỉ cần khai báo trong `supported_formats`).
- (Tùy chọn) Context về cuộc họp: tên cuộc họp, người tham dự dự kiến, agenda — dùng lại tham số `-c/--context` sẵn có để tăng độ chính xác transcript **và** làm input cho bước sinh biên bản họp.

### 3.2 Output
1. **Transcript**: `.txt` (đã có), `.srt` (cần thêm), `.json` (đã có, dùng làm input có cấu trúc cho bước tóm tắt).
2. **Meeting Minutes** (mới): file `.md` (và tuỳ chọn `.docx`) gồm:
   - Thông tin chung: tiêu đề cuộc họp, ngày giờ, thời lượng, người tham dự (suy ra từ speaker labels + context truyền vào).
   - Tóm tắt nội dung theo từng chủ đề/mục agenda.
   - Các quyết định đã chốt (decisions).
   - Action items: việc cần làm — người phụ trách (nếu xác định được) — hạn chót (nếu có nhắc tới).
   - Các vấn đề còn để ngỏ / cần theo dõi tiếp (open questions/follow-ups).
   - (Tuỳ chọn) Trích dẫn nguyên văn kèm timestamp cho các điểm quan trọng, để người đọc có thể tra lại transcript gốc.

---

## 4. Kiến trúc đề xuất

### 4.1 Pipeline tổng thể

```
[1] Audio → AudioProcessor → ChunkData[]                (tái dùng audio.py)
[2] ChunkData[] → TranscriptionEngine + LLMClient        (tái dùng transcriber.py, llm_client.py)
        → TranscriptionResult[] → merge/dedupe → Transcript (txt/srt/json)
[3] Transcript (json, có timestamp+speaker) + meeting_context
        → MinutesGenerator (LLM call MỚI, prompt tóm tắt có cấu trúc, output ép theo JSON schema)
        → MeetingMinutes (Pydantic model MỚI)
[4] MeetingMinutes → MinutesFormatter → .md / .docx / (tuỳ chọn gửi Slack/email)
```

### 4.2 Module mới cần xây

| Module mới | Trách nhiệm |
|---|---|
| `models.py` (mở rộng) | Thêm `MeetingMinutes`, `ActionItem`, `Decision`, `Participant`. |
| `minutes_generator.py` | Gọi LLM (text-only, không cần audio) trên **toàn bộ transcript đã gộp** (hoặc theo từng "topic segment" nếu transcript quá dài để vừa context window) để sinh JSON có cấu trúc theo schema `MeetingMinutes`. Có thể dùng `response_format={"type": "json_object"}` hoặc structured output của LiteLLM/Pydantic (`instructor`-style) để ép định dạng, tránh phải regex-parse như ở bước transcribe. |
| `minutes_formatter.py` | Render `MeetingMinutes` → Markdown đẹp (heading, bullet, bảng action items) → tuỳ chọn convert sang `.docx` (dùng `python-docx`). |
| `output.py` (mở rộng) | Thêm `write_srt_file`, `export_meeting_minutes`. |
| `cli.py` (mở rộng) | Thêm lệnh `llm-transcribe minutes audio.wav` (chạy full pipeline) hoặc flag `--minutes` trên lệnh hiện có; thêm `--minutes-model` để cho phép chọn model khác (rẻ hơn/mạnh hơn) cho bước tóm tắt so với bước transcribe. |

### 4.3 Vì sao tách 2 bước LLM (transcribe rồi mới summarize) thay vì 1 bước?

- Bước transcribe cần model **đa phương thức nghe được audio** (Gemini/GPT-4o/Claude); bước tóm tắt chỉ cần model **text giỏi suy luận/tổng hợp**, có thể dùng model rẻ hơn hoặc mạnh hơn tuỳ nhu cầu (ví dụ transcribe bằng `gemini-2.5-flash` cho rẻ, tóm tắt bằng model mạnh hơn để chất lượng biên bản tốt).
- Transcript đầy đủ (có thể rất dài với cuộc họp hàng giờ) khi đưa vào bước tóm tắt sẽ cần quản lý context window riêng — tách bước giúp dễ tối ưu (map-reduce theo từng đoạn nếu cần) mà không ảnh hưởng logic transcribe.
- Transcript là sản phẩm hữu ích độc lập (tra cứu lại nguyên văn); tách bước giúp người dùng chỉ cần transcript có thể dừng ở bước 1, không bắt buộc luôn phải tóm tắt.

### 4.4 Xử lý transcript dài khi tóm tắt (map-reduce nếu cần)

Nếu transcript vượt quá context window của model tóm tắt:
1. Chia transcript theo mốc thời gian lớn (ví dụ mỗi 30-45 phút) → tóm tắt từng đoạn ("partial minutes").
2. Gộp các partial minutes → 1 lần tóm tắt cuối ("reduce") để ra biên bản họp thống nhất, loại trùng lặp giữa các đoạn.
Đây là hướng mở rộng tương tự cách repo gốc xử lý context giữa các chunk audio, nhưng áp dụng ở tầng transcript/text thay vì audio.

---

## 5. Data model đề xuất (mở rộng `models.py`)

```python
class Participant(BaseModel):
    name: str
    role: Optional[str] = None  # vd: "Trưởng phòng", "PM"

class Decision(BaseModel):
    summary: str
    context: Optional[str] = None  # trích dẫn/timestamp liên quan

class ActionItem(BaseModel):
    description: str
    owner: Optional[str] = None
    due_date: Optional[str] = None
    status: str = "open"  # open | done | blocked

class MeetingMinutes(BaseModel):
    title: str
    meeting_date: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    participants: List[Participant] = []
    summary: str  # tóm tắt tổng quan
    topics: List[dict] = []  # [{ "heading": ..., "summary": ..., "timestamp_refs": [...] }]
    decisions: List[Decision] = []
    action_items: List[ActionItem] = []
    open_questions: List[str] = []
    model_used: str
    source_transcript_file: Path
```

---

## 6. Tech stack đề xuất

- **Giữ nguyên**: Python 3.13, Typer, Pydantic, pydub + FFmpeg, LiteLLM, stamina (retry), rich (CLI UX).
- **Thêm**:
  - `python-docx` — xuất biên bản họp dạng Word (nếu cần gửi cho người không quen Markdown).
  - Structured output: dùng khả năng "JSON mode"/function-calling của model qua LiteLLM (hoặc thư viện `instructor`) để ép LLM trả JSON đúng schema `MeetingMinutes`, tránh regex-parsing dễ vỡ như hiện tại.
  - (Tuỳ chọn, nếu cần giao diện) FastAPI + giao diện web đơn giản, hoặc tích hợp thành webhook nhận file ghi âm từ Zoom/Google Meet/MS Teams.
  - (Tuỳ chọn) Vector/keyword search trên transcript nếu về sau muốn "hỏi đáp" nội dung cuộc họp.

---

## 7. Roadmap đề xuất

1. **Giai đoạn 0 — Khảo sát & thử nghiệm**: fork/clone `llm-transcribe`, chạy thử với vài file audio mẫu (kể cả tiếng Việt) để đánh giá chất lượng transcript + tốc độ + chi phí thực tế trước khi đầu tư thêm.
2. **Giai đoạn 1 — Hoàn thiện transcript**: thêm export `.srt`; kiểm thử/tinh chỉnh prompt cho tiếng Việt nếu cần; cân nhắc xử lý song song nhiều chunk để giảm thời gian chờ với file dài.
3. **Giai đoạn 2 — Sinh Meeting Minutes**: xây `minutes_generator.py` + `MeetingMinutes` model + prompt tóm tắt có cấu trúc; xuất `.md`.
4. **Giai đoạn 3 — Hoàn thiện trải nghiệm**: xuất `.docx`, CLI command `minutes`, xử lý transcript dài (map-reduce), theo dõi chi phí riêng cho từng bước (transcribe vs. summarize).
5. **Giai đoạn 4 (tuỳ chọn)** — Giao diện/ tích hợp: web UI để upload file & xem kết quả, hoặc tích hợp tự động lấy file ghi âm từ Zoom/Google Meet, gửi biên bản họp qua email/Slack.

---

## 8. Câu hỏi cần làm rõ với người dùng (open questions)

- Ngôn ngữ chính của các cuộc họp là tiếng Việt, tiếng Anh, hay cả hai? (ảnh hưởng tới việc tinh chỉnh prompt và chọn model).
- Cuộc họp thường dài bao lâu? (ảnh hưởng chiến lược chunking/map-reduce và ước tính chi phí).
- Có yêu cầu chạy hoàn toàn on-premise/không gửi dữ liệu ra ngoài (do tính bảo mật cuộc họp nội bộ) hay chấp nhận dùng API LLM bên thứ 3 (Gemini/OpenAI/Anthropic)?
- Đầu ra biên bản họp cần theo mẫu/template cụ thể của công ty không (ví dụ mẫu biên bản họp chuẩn hoá sẵn)?
- Người dùng cuối là dev (dùng CLI ổn) hay cần giao diện web/app để dùng?

---

## 9. Nguồn tham khảo

- Repo: https://github.com/rivol/llm-transcribe
- File đã đọc trực tiếp từ repo: `README.md`, `docs/architecture.md`, `pyproject.toml`, `main.py`, `src/llm_transcribe/{cli,models,audio,llm_client,transcriber,output,timestamp_utils}.py`.
