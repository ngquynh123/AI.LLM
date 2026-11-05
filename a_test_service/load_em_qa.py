"""
Script chạy để sinh embedding cho tập QA/dialogue (EmbedQA service).

Hướng dẫn nhanh (từ thư mục gốc của repo):
    # Chạy nhanh dùng embedding giả (không cần cài model)
    & .\.venv\Scripts\python.exe .\test_and_services\a_test_service\load_em_qa.py data\dialogue\CTSC.json --dummy

    # Chạy thực tế (cần cài sentence-transformers và torch)
    & .\.venv\Scripts\python.exe .\test_and_services\a_test_service\load_em_qa.py data\dialogue\CTSC.json

Kết quả:
    - File đầu ra: embed_data/dialogue/<tên_file>_with_emb.json

Ghi chú:
    - Nếu script in "Usage: ..." nghĩa là bạn chưa truyền đường dẫn file JSON.
    - Dùng `--dummy` để test nhanh khi chưa cài model.
"""
import sys
from pathlib import Path
import json

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from test_and_services.services.embed_qa_service import EmbedQAService


def main():
    if len(sys.argv) < 2:
        print("Usage: python load_em_qa.py <path_to_qa_json>")
        return
    in_path = Path(sys.argv[1])
    if not in_path.exists():
        print(f"Input not found: {in_path}")
        return

    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    use_dummy = "--dummy" in sys.argv
    svc = EmbedQAService(use_dummy=use_dummy)
    out = svc.embed_qa_items(data)

    out_dir = Path("embed_data/dialogue")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{in_path.stem}_with_emb.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Done: wrote {out_path}")


if __name__ == "__main__":
    main()
