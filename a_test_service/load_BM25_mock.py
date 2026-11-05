# -*- coding: utf-8 -*-
"""
Script kiểm thử cho bộ nạp BM25 (`service/load_data_BM25.py`).
Chạy trong `.venv` của dự án để kiểm tra việc nạp dữ liệu và in ra một mẫu nhỏ.
"""
import sys
from pathlib import Path

# Ensure repo root is importable
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from services.load_data_BM25_mock import load_data_bm25


def main():
    try:
        records, docs = load_data_bm25()
        print(f"Loaded records: {len(records)}  docs: {len(docs)}")
        # đếm số bản ghi theo kênh
        counts = {}
        for r in records:
            ch = r.get('channel','unknown')
            counts[ch] = counts.get(ch,0) + 1
        for ch, c in sorted(counts.items()):
            print(f"  {ch}: {c}")
        if records:
            print('\nSample record 0:')
            import json
            print(json.dumps(records[0], ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"ERROR: {e}")
        return 2
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
