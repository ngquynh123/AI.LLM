# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import json
import numpy as np

BEST_TEXT_KEYS = ("search_text", "text", "content", "answer_text")

def _read_json(p: Path):
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        if "items" in data and isinstance(data["items"], list): return data["items"]
        if "data"  in data and isinstance(data["data"], list):  return data["data"]
        return [data]
    return data

def _best_text(d: Dict[str, Any]) -> str:
    for k in BEST_TEXT_KEYS:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    title = d.get("title") or d.get("tittle")
    q = d.get("question"); a = d.get("answer") or d.get("answer_text")
    if isinstance(q, str) and isinstance(a, str) and q.strip() and a.strip():
        base = (q.strip() + "\n" + a.strip()).strip()
        return (f"{title.strip()}\n{base}".strip()
                if isinstance(title, str) and title.strip() else base)
    if isinstance(q, str) and q.strip():
        return (f"{title.strip()}\n{q.strip()}".strip()
                if isinstance(title, str) and title.strip() else q.strip())
    if isinstance(title, str) and title.strip():
        return title.strip()
    return (d.get("text") or "").strip()

def _meta_keep(d: Dict[str, Any]) -> Dict[str, Any]:
    keep_keys = [
        "_id","id","doc_id","subject","subject_id","topic","grade","lesson",
        "lesson_id","source","file","chunk_id","title","tittle","order","importance"
    ]
    return {k: d.get(k) for k in keep_keys if k in d}

def _extract_qa(d: Dict[str, Any]):
    q = d.get("question")
    a = d.get("answer") or d.get("answer_text")
    q = q.strip() if isinstance(q, str) and q.strip() else None
    a = a.strip() if isinstance(a, str) and a.strip() else None
    return q, a

def load_corpus(root: str="embed_data", channels=("dialogue","lesson")) -> Tuple[np.ndarray, List[Dict[str,Any]]]:
    """
    Chỉ nạp CORPUS ĐÃ EMBED trong embed_data/{dialogue,lesson}
    (qa600.json sẽ được xử lý riêng bằng BM25 trong main.py)
    """
    data_root = Path(root)
    vectors: List[np.ndarray] = []
    records: List[Dict[str,Any]] = []

    for ch in channels:
        ch_dir = data_root / ch
        if not ch_dir.exists():
            print(f"[WARN] Missing channel dir: {ch_dir}")
            continue
        for p in sorted(ch_dir.glob("*.json")):
            items = _read_json(p)
            for it in items:
                # Case A: 1 record có sẵn embedding
                emb = it.get("embedding") if it.get("embedding") is not None else it.get("embed")
                if isinstance(emb, list):
                    vec = np.asarray(emb, dtype=np.float32)
                    q, a = _extract_qa(it)
                    records.append({
                        "text": _best_text(it),
                        "meta": _meta_keep(it),
                        "file": p.name,
                        "channel": ch,
                        "qa": {"question": q, "answer": a} if (q or a) else None,
                    })
                    vectors.append(vec)
                    continue
                # Case B: bản ghi lesson có chunks[]
                if isinstance(it.get("chunks"), list):
                    base_meta = _meta_keep(it)
                    for idx, ck in enumerate(it["chunks"], 1):
                        emb2 = ck.get("embedding") if ck.get("embedding") is not None else ck.get("embed")
                        if not isinstance(emb2, list):
                            continue
                        vec2 = np.asarray(emb2, dtype=np.float32)
                        meta = {**base_meta, **_meta_keep(ck), "chunk_index": idx}
                        text = _best_text({**it, **ck}) or ck.get("text","")
                        records.append({
                            "text": text,
                            "meta": meta,
                            "file": p.name,
                            "channel": ch,
                            "qa": None,
                        })
                        vectors.append(vec2)

    if not vectors:
        raise RuntimeError("No vectors found in embed_data. Mỗi record/chunk cần có 'embedding' hoặc 'embed'.")

    mat = np.vstack(vectors).astype(np.float32)
    for i, r in enumerate(records):
        r["rid"] = i
    print(f"[OK] Loaded {len(records)} embedded records from {list(channels)} | dim={mat.shape[1]}")
    return mat, records
