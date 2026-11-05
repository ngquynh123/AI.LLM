"""
Script test nhỏ để sinh embedding cho dữ liệu bài học (lesson).

Hướng dẫn nhanh (từ thư mục gốc của repo):
    # Chạy nhanh dùng embedding giả (không cần cài model)
    & .\.venv\Scripts\python.exe .\test_and_services\a_test_service\load_em_lesson.py data\lesson\CTSC.json --dummy

    # Chạy thật (cần cài sentence-transformers và torch)
    & .\.venv\Scripts\python.exe .\test_and_services\a_test_service\load_em_lesson.py data\lesson\CTSC.json

Kết quả:
    - File đầu ra: embed_data/lesson/<tên_file>_with_emb.json

Ghi chú:
    - Nếu script in "Usage: ..." nghĩa là bạn chưa truyền đường dẫn file JSON.
    - Dùng `--dummy` khi muốn test nhanh mà không cài model.
"""
import sys
from pathlib import Path
import json

# Đảm bảo thư mục gốc của repo có trong sys.path để các import hoạt động khi chạy script trực tiếp
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from test_and_services.services.em_lesson_service import EmbeddingService


def main():
    if len(sys.argv) < 2:
        print("Usage: python load_em_lesson.py <path_to_lesson_json>")
        return
    in_path = Path(sys.argv[1])
    if not in_path.exists():
        print(f"Input not found: {in_path}")
        return

    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Hỗ trợ cờ --dummy cho môi trường không có sentence-transformers
    use_dummy = "--dummy" in sys.argv
    svc = EmbeddingService(use_dummy=use_dummy)
    out = svc.embed_documents(data)

    out_dir = Path("embed_data/lesson")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{in_path.stem}_with_emb.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Done: wrote {out_path}")


if __name__ == "__main__":
    main()
