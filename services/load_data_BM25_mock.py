"""Bộ nạp dữ liệu độc lập cho BM25.

Tập lệnh này đọc các file JSON từ:
- data/dialogue/*.json
- data/lesson/*.json
- một file QA chuẩn tại data/qa600.json (nếu có). Nếu không có, sẽ dò fallback
    sang data/qa600_sub.json hoặc các file khớp mẫu qa600*.json

Trả về hai danh sách:
- records: danh sách dict, mỗi mục có 'channel', 'file' và 'qa' hoặc 'text'
- docs: danh sách chuỗi (text) dùng để xây chỉ mục BM25

Chỉ dùng thư viện chuẩn của Python; không import module nội bộ của dự án.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _extract_q_and_a_from_item(it: Any) -> Tuple[Any, Any]:
    if not isinstance(it, dict):
        return None, None
    if 'qa' in it and isinstance(it['qa'], dict):
        q = it['qa'].get('question') or it['qa'].get('q')
        a = it['qa'].get('answer') or it['qa'].get('a')
        return q, a
    q = it.get('question') or it.get('q') or it.get('prompt')
    a = it.get('answer') or it.get('a') or it.get('completion')
    return q, a


def _load_channel(folder: Path, channel: str, records: List[Dict[str, Any]], docs: List[str]) -> None:
    if not folder.exists() or not folder.is_dir():
        return
    for path in sorted(folder.glob('*.json')):
        try:
            raw = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            continue
        items = raw if isinstance(raw, list) else [raw]
        for it in items:
            rec: Dict[str, Any] = {'channel': channel, 'file': path.name}
            q, a = _extract_q_and_a_from_item(it)
            if q or a:
                rec['qa'] = {'question': q, 'answer': a}
                docs.append('\n'.join([str(x).strip() for x in [q, a] if x]))
                # copy over common meta keys if present (lesson, subject, ids)
                if isinstance(it, dict):
                    for k in ('lesson', 'lesson_id', 'subject', 'subject_id', 'topic', '_id', 'id'):
                        if k in it:
                            rec[k] = it.get(k)
            else:
                    # handle lesson-style files which may store content in 'chunks'
                    txt = ''
                    # common title fields
                    title = (it.get('tittle') or it.get('title') or '').strip() if isinstance(it, dict) else ''
                    if isinstance(it, dict):
                        # join chunk texts if present
                        chunks = it.get('chunks')
                        if isinstance(chunks, list) and chunks:
                            pieces = []
                            for c in chunks:
                                if isinstance(c, dict):
                                    t = c.get('text') or c.get('content') or ''
                                    if t:
                                        pieces.append(str(t).strip())
                                elif isinstance(c, str):
                                    pieces.append(c.strip())
                            txt = "\n\n".join(pieces)
                    # fallback to text/content fields if no chunks found
                    if not txt:
                        txt = (it.get('text') or it.get('content') or '').strip() if isinstance(it, dict) else ''
                    # prepend title when available
                    if title:
                        txt = (title + "\n\n" + txt) if txt else title
                    # copy over common meta keys if present (lesson, subject, ids)
                    if isinstance(it, dict):
                        for k in ('lesson', 'lesson_id', 'subject', 'subject_id', 'topic', '_id', 'id'):
                            if k in it:
                                rec[k] = it.get(k)
                    rec['text'] = txt
                    docs.append(str(txt).strip())
            records.append(rec)


def _load_qa600_file(path: Path, records: List[Dict[str, Any]], docs: List[str]) -> None:
    if not path.exists():
        return
    try:
        items = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return
    # Nếu file chứa 1 object duy nhất, bọc thành list để xử lý đồng nhất
    if isinstance(items, dict):
        items = [items]
    for it in items:
        q = (it.get('question') or '').strip() if isinstance(it, dict) else ''
        a = (it.get('answer') or it.get('answer_text') or '').strip() if isinstance(it, dict) else ''
        # Bỏ qua mục không có câu hỏi và không có đáp án
        if not (q or a):
            continue
        rec = {'channel': 'qa600', 'file': path.name, 'qa': {'question': q or None, 'answer': a or None}}
        records.append(rec)
        docs.append('\n'.join([x for x in [q, a] if x]))


def load_data_bm25(data_root: str | Path = 'data', channels: List[str] | None = None, qa600_path: str | Path | None = None) -> Tuple[List[Dict[str, Any]], List[str]]:
    data_root = Path(data_root)
    if channels is None:
        channels = ['dialogue', 'lesson']

    records: List[Dict[str, Any]] = []
    docs: List[str] = []

    for ch in channels:
        _load_channel(data_root / ch, ch, records, docs)

    # Tải các file qa600 (nếu có): nếu truyền qa600_path thì dùng file đó,
    # còn không thì tự động dò các file khớp mẫu qa600*.json trong thư mục data
    if qa600_path:
        _load_qa600_file(Path(qa600_path), records, docs)
    else:
        # Prefer a single canonical qa600.json first; fallback to qa600_sub.json or any qa600*.json
        preferred = data_root / 'qa600.json'
        if preferred.exists():
            _load_qa600_file(preferred, records, docs)
        else:
            alt = data_root / 'qa600_sub.json'
            if alt.exists():
                _load_qa600_file(alt, records, docs)
            else:
                for p in sorted(data_root.glob('qa600*.json')):
                    _load_qa600_file(p, records, docs)

    return records, docs


def _cli(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='Tải dữ liệu cho BM25 (dialogue, lesson, qa600)')
    ap.add_argument('--data-root', type=str, default='data')
    ap.add_argument('--channels', nargs='+', default=['dialogue', 'lesson'])
    ap.add_argument('--qa600-path', type=str, default=None)
    ap.add_argument('--out-records', type=str, default=None)
    ap.add_argument('--out-docs', type=str, default=None)
    args = ap.parse_args(argv)

    records, docs = load_data_bm25(args.data_root, args.channels, qa600_path=args.qa600_path)
    print(f'Loaded records={len(records)} docs={len(docs)}')
    ch_counts = {}
    for r in records:
        ch = r.get('channel', 'unknown')
        ch_counts[ch] = ch_counts.get(ch, 0) + 1
    for ch, cnt in sorted(ch_counts.items()):
        print(f'  {ch}: {cnt} records')

    # Additional: report qa600_sub or other qa600 file counts
    data_root_path = Path(args.data_root)
    if args.qa600_path:
        qa_paths = [Path(args.qa600_path)]
    else:
        preferred = data_root_path / 'qa600.json'
        if preferred.exists():
            qa_paths = [preferred]
        else:
            alt = data_root_path / 'qa600_sub.json'
            if alt.exists():
                qa_paths = [alt]
            else:
                qa_paths = sorted(data_root_path.glob('qa600*.json'))

    total_qa_records = sum(1 for r in records if r.get('channel') == 'qa600')
    if qa_paths:
        print('\nqa600 files considered:')
        for p in qa_paths:
            fname = p.name
            cnt = sum(1 for r in records if r.get('channel') == 'qa600' and r.get('file') == fname)
            print(f'  {fname}: {cnt} records')
        print(f'  total qa600 records: {total_qa_records}')
    else:
        print('\nNo qa600 files found in', str(data_root_path))

    if args.out_records:
        Path(args.out_records).write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
        print('Wrote', args.out_records)
    if args.out_docs:
        Path(args.out_docs).write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding='utf-8')
        print('Wrote', args.out_docs)

    return 0


# if __name__ == '__main__':
#     raise SystemExit(_cli())
