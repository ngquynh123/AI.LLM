# AI.LLM

AI.LLM là chatbot hỏi đáp tiếng Việt, tập trung vào nội dung lý thuyết lái xe. Hệ thống kết hợp:

- **Semantic search** bằng embedding và FAISS.
- **BM25** để tìm kiếm từ khóa, đặc biệt cho dữ liệu câu hỏi - đáp án.
- **LLM qua Ollama** để tạo câu trả lời dựa trên ngữ cảnh đã truy xuất.
- **FastAPI** để cung cấp API truy vấn, health check và load test.

## Kiến trúc

```text
POST /query
    │
    ├── Tiền xử lý câu hỏi
    ├── Semantic search trên embed_data/{dialogue,lesson}
    ├── BM25 search trên corpus và data/qa600.json
    ├── Hợp nhất ngữ cảnh
    └── Ollama sinh câu trả lời
```

Các thư mục chính:

| Thư mục | Mục đích |
| --- | --- |
| `retriever/` | API, semantic search, BM25, prompt và memory |
| `services/` | Các service xử lý dữ liệu, kết quả và embedding |
| `tools/` | Công cụ chạy batch/load test |
| `tests/` | Test tự động |
| `models/` | Model/tokenizer hoặc artifact phục vụ local |
| `embed_data/` | Corpus đã tạo embedding, không lưu trong Git |
| `data/` | Dữ liệu đầu vào, gồm `data/qa600.json` |

## Yêu cầu

- Python 3.10 trở lên
- Ollama đang chạy tại `http://localhost:11434` (mặc định)
- Model Ollama có tên `phogpt4b-ft` (hoặc tên được cấu hình qua `LLM_MODEL_ID`)
- Đủ dung lượng cho model và dữ liệu embedding

`faiss-cpu` có thể gặp vấn đề khi cài trên Windows. Nếu cài đặt native không thành công, nên dùng WSL2 hoặc Conda.

## Cài đặt

Từ thư mục gốc repository:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nếu PowerShell chặn activate script, có thể chạy lệnh cài đặt trực tiếp bằng:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Chuẩn bị Ollama

Cài Ollama theo hướng dẫn tại <https://ollama.com/download>, khởi động Ollama rồi kiểm tra model:

```powershell
ollama list
ollama pull phogpt4b-ft
```

Model trong `models/Modelfile.txt` có thể được dùng làm cơ sở để tạo model Ollama:

```powershell
ollama create phogpt4b-ft -f models\Modelfile.txt
```

Nếu Ollama chạy ở địa chỉ khác, đặt biến môi trường:

```powershell
$env:LLM_OLLAMA_HOST = "http://localhost:11434"
$env:LLM_MODEL_ID = "phogpt4b-ft"
```

## Khôi phục artifact lớn

Các artifact lớn không được theo dõi trong Git. Hệ thống cần tối thiểu:

```text
embed_data/
data/qa600.json
```

Tùy cách chạy model, có thể cần thêm:

```text
models/
out_lora_adapter/
PL_dialogue/PLGTDB_emb.json
```

Copy artifact từ storage nội bộ hoặc máy lưu trữ vào đúng đường dẫn trong repository. Xem [LARGE_ARTIFACTS_COPY_INSTRUCTIONS.md](LARGE_ARTIFACTS_COPY_INSTRUCTIONS.md) để biết chi tiết. Không commit file `.safetensors`, embedding lớn hoặc model binary vào Git.

`embed_data/dialogue/` và `embed_data/lesson/` phải chứa các file JSON có trường `embedding` hoặc `embed`. Nếu không tìm thấy vector, server sẽ dừng khi khởi động.

## Chạy API

Chạy từ thư mục gốc repository:

```powershell
python -m uvicorn retriever.api_server:app --host 0.0.0.0 --port 8001 --reload
```

Kiểm tra server:

```powershell
Invoke-RestMethod http://127.0.0.1:8001/health
```

Kết quả mong đợi:

```json
{"status":"ok"}
```

### Gửi câu hỏi

```powershell
$body = @{
    question = "Vượt xe trong hầm đường bộ có được phép hay không?"
    session_id = "demo"
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri http://127.0.0.1:8001/query `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

Request `/query` hỗ trợ:

| Trường | Bắt buộc | Mặc định | Mô tả |
| --- | --- | --- | --- |
| `question` | Có | - | Câu hỏi cần trả lời |
| `session_id` | Không | `api` | Định danh phiên hội thoại |
| `debug` | Không | `false` | Bật thông tin debug |
| `debug_level` | Không | `normal` | `minimal`, `normal` hoặc `detailed` |

Các endpoint khác:

- `GET /health`: kiểm tra trạng thái server.
- `GET /load-results`: đọc kết quả load test từ file JSONL.
- `POST /load-test/async-run`: chạy load test bất đồng bộ từ file CSV câu hỏi.

Có thể mở tài liệu OpenAPI tại <http://127.0.0.1:8001/docs>.

## Cấu hình môi trường

Các biến thường dùng:

| Biến | Mặc định | Mô tả |
| --- | --- | --- |
| `LLM_OLLAMA_HOST` | `http://localhost:11434` | Địa chỉ Ollama |
| `LLM_MODEL_ID` | `phogpt4b-ft` | Tên model Ollama |
| `LLM_TEMP` | `0.3` | Temperature khi sinh câu trả lời |
| `LLM_TIMEOUT_CONNECT` | `10` | Timeout kết nối, tính bằng giây |
| `LLM_TIMEOUT_READ` | `600` | Timeout đọc response, tính bằng giây |
| `LLM_KEEP_ALIVE` | `5m` | Thời gian giữ model trong Ollama |
| `EMB_MODEL` | `AITeamVN/Vietnamese_Embedding` | Model embedding |
| `MAX_CONCURRENT_QUERIES` | `4` | Số truy vấn đồng thời tối đa |
| `QUERY_QUEUE_TIMEOUT` | `600` | Thời gian chờ queue, tính bằng giây |
| `LLM_ENABLE_MEMORY` | `0` | Bật memory chain khi đặt `1` hoặc `true` |
| `CONV_MEMORY_DEPTH` | `4` | Số lượt hội thoại dùng cho memory |

Server mặc định bật chế độ offline của Hugging Face (`HF_HUB_OFFLINE=1` và `TRANSFORMERS_OFFLINE=1`). Các model embedding cần có sẵn trong cache nếu môi trường không có Internet.

## Chạy test

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Chạy một nhóm test:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_determinism.py -q
```

## Load test

Chuẩn bị file CSV có cột `question`, sau đó gọi endpoint:

```powershell
$body = @{
    url = "http://127.0.0.1:8001/query"
    questions = "questions.csv"
    total = 100
    concurrency = 20
    save_responses = "out/load.jsonl"
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri http://127.0.0.1:8001/load-test/async-run `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

Kết quả mặc định được lưu tại `out/load.jsonl`; có thể đọc lại bằng:

```powershell
Invoke-RestMethod "http://127.0.0.1:8001/load-results?path=out/load.jsonl"
```

## Xử lý lỗi thường gặp

- **`No vectors found in embed_data`**: kiểm tra `embed_data/dialogue/` và `embed_data/lesson/`, cũng như trường `embedding`/`embed` trong JSON.
- **Không kết nối được Ollama**: kiểm tra Ollama đang chạy, `LLM_OLLAMA_HOST` và model trong `ollama list`.
- **Không tìm thấy model embedding**: tải model trước hoặc tắt chế độ offline khi cần tải từ Hugging Face.
- **Không cài được `faiss-cpu` trên Windows**: dùng WSL2 hoặc Conda.
- **Thiếu file `data/qa600.json`**: khôi phục dữ liệu đầu vào theo phần artifact ở trên.

## Lưu ý Git

Model, embedding và các file sinh ra khi load test không nên commit. Kiểm tra `.gitignore` trước khi chạy `git add`; nếu artifact lớn đã từng được commit, cần xử lý lịch sử Git hoặc chuyển sang Git LFS/external storage trước khi push.
