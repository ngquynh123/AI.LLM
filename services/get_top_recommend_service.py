"""Lấy TOP N gợi ý QA từ kết quả BM25 và in ra dưới dạng hộp (box) thân thiện.

Hàm chính: get_top_recommend(records, bm25_hits, topn=3)
- records: list[dict] (mỗi record giống cấu trúc trong retriever/main.py)
- bm25_hits: list[dict] (mỗi mục có 'rid' và optional metadata như score)
- topn: số lượng gợi ý (mặc định 3)

CLI:
- python service/get_top_recommend.py --records records.json --hits hits.json
  records.json: danh sách record (một file JSON)
  hits.json: danh sách hit (một file JSON, mỗi entry có trường 'rid')
"""

from __future__ import annotations

import json
import textwrap
from typing import List, Dict, Any


def _term_width(default: int = 80) -> int:
    """Trả về chiều rộng terminal hiện tại (dùng để vẽ box)."""
    try:
        import shutil as _sh
        return max(40, _sh.get_terminal_size((default, 20)).columns)
    except Exception:
        return default


def _box(title: str, body: str, width: int = None) -> str:
    """Tạo chuỗi "box" đẹp để in ra terminal với tiêu đề và nội dung.

    Dùng ký tự khung Unicode để hiển thị rõ ràng trên terminal.
    """
    width = width or _term_width()
    line = "═" * (width - 2)
    top, bot = f"╔{line}╗", f"╚{line}╝"
    head, sep = f"║ {title.strip()}".ljust(width - 1) + "║", "╟" + ("─" * (width - 2)) + "╢"
    rows = []
    for paragraph in (body or "").split("\n"):
        if not paragraph.strip():
            rows.append("║".ljust(width - 1) + "║")
            continue
        for w in (textwrap.wrap(paragraph, width=width - 4) or [""]):
            rows.append(("║ " + w).ljust(width - 1) + "║")
    return "\n".join([top, head, sep] + rows + [bot])


def _extract_q_and_a(r: Dict[str, Any]):
    """Lấy câu hỏi và câu trả lời từ một record.

    Hỗ trợ hai dạng: record có key 'qa' (object với question/answer) hoặc trực tiếp 'question'/'answer'.
    Trả về tuple (q, a) với giá trị None nếu không có.
    """
    qa = r.get("qa") or {}
    q = qa.get("question") or r.get("question")
    a = qa.get("answer") or r.get("answer")
    q = q.strip() if isinstance(q, str) and q.strip() else None
    a = a.strip() if isinstance(a, str) and a.strip() else None
    return q, a


def get_top_recommend(records: List[Dict[str, Any]], bm25_hits: List[Dict[str, Any]], topn: int = 3) -> str:
    """Trả về chuỗi hiển thị TOP-N QA gợi ý từ `bm25_hits`.

    Mỗi `hit` trong `bm25_hits` phải chứa 'rid' (index vào `records`).
    """
    """Tạo phần hiển thị TOP-N gợi ý.

    Lấy lần lượt các hit từ `bm25_hits` (theo thứ tự đã sắp) và trích xuất câu hỏi/đáp tương ứng
    từ `records`. Kết quả trả về là một chuỗi đã được định dạng sẵn (box).
    """
    lines = []
    for i, h in enumerate(bm25_hits[:topn], 1):
        rid = h.get("rid")
        if rid is None or not (0 <= rid < len(records)):
            continue
        r = records[rid]
        q, a = _extract_q_and_a(r)
        header = f"#{i:02d} [{r.get('channel','?')}/{r.get('file','?')}]"
        lines.append(header)
        if q:
            prev_q = q.replace("\n", " ")
            if len(prev_q) > 200: prev_q = prev_q[:200] + "…"
            lines.append(f"  Q: {prev_q}")
        if a:
            prev_a = a.replace("\n", " ")
            if len(prev_a) > 200: prev_a = prev_a[:200] + "…"
            lines.append(f"  A: {prev_a}")
        if not (q or a):
            txt = (r.get("text") or "").replace("\n", " ")
            if len(txt) > 200: txt = txt[:200] + "…"
            lines.append(f"  {txt}")

    body = "\n".join(lines) if lines else "(không có)"
    return _box("TOP {} QA GỢI Ý".format(min(topn, len(lines) or topn)), body)


def _cli():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, help="Path to records JSON")
    ap.add_argument("--hits", required=True, help="Path to hits JSON (list of {'rid': int})")
    ap.add_argument("--topn", type=int, default=3)
    args = ap.parse_args()

    with open(args.records, "r", encoding="utf-8") as f:
        records = json.load(f)
    with open(args.hits, "r", encoding="utf-8") as f:
        hits = json.load(f)

    print(get_top_recommend(records, hits, topn=args.topn))


if __name__ == "__main__":
    _cli()
