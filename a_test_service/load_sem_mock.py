# -*- coding: utf-8 -*-
"""
Script kiểm thử: nạp embeddings semantic bằng `service.load_data_sematic`.

Cách chạy (PowerShell):
    & .venv/Scripts/python.exe AI_test_service/load_sem_mock.py

Script này sẽ đảm bảo thư mục gốc repository được thêm vào `sys.path` để
`from service.load_data_sematic import load_corpus` hoạt động khi chạy trực tiếp.
"""
import sys
from pathlib import Path

# Đảm bảo thư mục gốc của repo nằm trên sys.path để có thể import `service` khi chạy
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from services.load_data_semantic_mock import load_corpus


def main():
    try:
        mat, records = load_corpus()
        print(f"Loaded matrix: shape={mat.shape}, dtype={mat.dtype}")
        print(f"Records: {len(records)}")
        if records:
            print("Sample record 0:")
            r = records[0]
            for k in ["rid", "channel", "file", "text"]:
                print(f"  {k}: {r.get(k)!s}")
    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == '__main__':
    main()
