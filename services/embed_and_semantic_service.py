# services/embed_and_semantic.py
# -*- coding: utf-8 -*-
"""
Service: Embed_and_sematic
Input : { "message": str, "top_k"?: int }
Output: { "metadata": [ { "text": str, "sem_thres": float } ] }

- Không kèm corpus mẫu. Phải set_corpus_provider(...) trước khi run().
- Có thể set_embedder(...) để dùng embedder ngoài (VD: ViBGE/BGE-M3).
- Nếu không set embedder, service tự build TF-IDF (IDF từ corpus) và dùng cosine.
- sem_thres là điểm cosine đã chuẩn hoá min-max về [0, 1].
"""
from __future__ import annotations
from typing import List, Dict, Tuple, Callable, Iterable, Optional, Union, Any
from collections import Counter
import math, re, unicodedata

Vector = Union[Dict[str, float], List[float]]

# ============ Normalize & tokenize ============
_TOKEN = re.compile(r"[A-Za-zÀ-ỹ0-9_]+")

def _normalize(text: str) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFC", text).strip().lower()
    return re.sub(r"\s+", " ", t)

def _tok(text: str) -> List[str]:
    return [w for w in _TOKEN.findall(text)]

# ============ Injection points ============
_CORPUS_PROVIDER: Optional[Callable[[], Iterable[str]]] = None
_EMBEDDER: Optional[Callable[[str], Vector]] = None

# ============ Caches (rebuilt on provider/embedder change) ============
_CORPUS_TEXTS: List[str] = []
_IDF: Dict[str, float] = {}
_VECS: List[Vector] = []         # song song với _CORPUS_TEXTS
_CORPUS_METAS: List[Dict[str, Any]] = []
_TFIDF_READY: bool = False

def set_corpus_provider(provider: Callable[[], Iterable[str]]) -> None:
    """Đăng ký nguồn corpus (Iterable[str])."""
    global _CORPUS_PROVIDER, _CORPUS_TEXTS, _IDF, _VECS, _TFIDF_READY, _CORPUS_METAS
    _CORPUS_PROVIDER = provider
    _CORPUS_TEXTS, _IDF, _VECS, _CORPUS_METAS = [], {}, [], []
    _TFIDF_READY = False

def set_embedder(embedder: Callable[[str], Vector]) -> None:
    """Đăng ký embedder ngoài. Đổi embedder sẽ buộc rebuild vector corpus."""
    global _EMBEDDER, _VECS, _TFIDF_READY
    _EMBEDDER = embedder
    _VECS = []
    _TFIDF_READY = False

# ============ Cosine helpers ============
def _cos_dict(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = 0.0
    for k, va in a.items():
        vb = b.get(k)
        if vb is not None:
            dot += va * vb
    na = math.sqrt(sum(v*v for v in a.values()))
    nb = math.sqrt(sum(v*v for v in b.values()))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)

def _cos_list(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)

def _cosine(a: Vector, b: Vector) -> float:
    if isinstance(a, dict) and isinstance(b, dict):
        return _cos_dict(a, b)
    if isinstance(a, list) and isinstance(b, list):
        return _cos_list(a, b)
    return 0.0

# ============ TF-IDF fallback ============
def _build_tfidf_if_needed() -> None:
    """Build IDF + TF-IDF vectors nếu không dùng embedder ngoài."""
    global _TFIDF_READY, _CORPUS_TEXTS, _IDF, _VECS
    if _TFIDF_READY:
        return
    if _CORPUS_PROVIDER is None:
        _CORPUS_TEXTS, _IDF, _VECS = [], {}, []
        _TFIDF_READY = True
        return

    texts = [t for t in _CORPUS_PROVIDER() if isinstance(t, str)]
    _CORPUS_TEXTS = texts

    docs = [set(_tok(_normalize(t))) for t in texts]
    N = len(docs)
    df = Counter()
    for d in docs:
        df.update(d)
    _IDF = {w: math.log(1 + (N - dfw + 0.5) / (dfw + 0.5)) for w, dfw in df.items()} if N > 0 else {}

    vecs: List[Dict[str, float]] = []
    for t in texts:
        toks = _tok(_normalize(t))
        tf = Counter(toks)
        vecs.append({w: tf[w] * _IDF.get(w, 0.0) for w in tf})
    _VECS = vecs
    _TFIDF_READY = True

def _embed_query_default(message: str) -> Dict[str, float]:
    toks = _tok(_normalize(message))
    tf = Counter(toks)
    return {w: tf[w] * _IDF.get(w, 0.0) for w in tf} if _IDF else {}

def _ensure_corpus_vectors() -> None:
    """Chuẩn bị vectors cho corpus theo embedder hiện tại."""
    global _CORPUS_TEXTS, _VECS
    if _EMBEDDER is None:
        _build_tfidf_if_needed()
        return

    # Có embedder ngoài
    if _CORPUS_PROVIDER is None:
        _CORPUS_TEXTS, _VECS[:] = [], []
        return
    texts = [t for t in _CORPUS_PROVIDER() if isinstance(t, str)]
    # allow provider to yield either str or (text, meta) tuples or dicts with text/meta
    items = list(_CORPUS_PROVIDER())
    texts = []
    metas = []
    for it in items:
        if isinstance(it, str):
            texts.append(it)
            metas.append({})
        elif isinstance(it, dict):
            txt = it.get("text") or it.get("search_text") or it.get("content") or it.get("answer_text") or ""
            texts.append(txt)
            metas.append({k: v for k, v in it.items() if k not in ("text", "embedding", "embed")})
        elif isinstance(it, (list, tuple)) and len(it) >= 1:
            texts.append(it[0])
            metas.append(it[1] if len(it) > 1 and isinstance(it[1], dict) else {})
        else:
            texts.append(str(it))
            metas.append({})

    if _VECS and len(_VECS) == len(texts):
        _CORPUS_TEXTS = texts
        _CORPUS_METAS = metas
        return
    _CORPUS_TEXTS = texts
    _CORPUS_METAS = metas
    # obtain embeddings and normalize them to unit vectors (list of floats)
    raw_vecs = [_EMBEDDER(t) for t in texts] if texts else []
    normed: List[Vector] = []
    for v in raw_vecs:
        if v is None:
            normed.append([])
            continue
        # convert numpy arrays to lists if needed
        try:
            # numpy array path
            import numpy as _np
            if _np is not None and hasattr(v, 'astype'):
                arr = _np.array(v, dtype=_np.float32).astype(_np.float32)
                n = _np.linalg.norm(arr)
                if n == 0:
                    normed.append(arr.tolist())
                else:
                    normed.append((arr / n).tolist())
                continue
        except Exception:
            pass
        # list/tuple path
        if isinstance(v, (list, tuple)):
            arr = [float(x) for x in v]
            norm = math.sqrt(sum(x*x for x in arr))
            if norm == 0:
                normed.append(arr)
            else:
                normed.append([x / norm for x in arr])
            continue
        # fallback: keep as-is
        normed.append(v)
    _VECS = normed

# ============ Utilities ============
def _minmax(scores: List[float]) -> List[float]:
    if not scores:
        return []
    mn, mx = min(scores), max(scores)
    if mx - mn == 0:
        return [1.0] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]

# ============ Service entry ============
def run(input_data: Dict) -> Dict:
    """
    Input : { "message": str, "top_k"?: int }
    Output: { "metadata": [ { "text": str, "sem_thres": float } ] }
    """
    msg = (input_data or {}).get("message", "") or ""
    if not msg.strip():
        return {"metadata": []}

    top_k = int((input_data or {}).get("top_k", 10))
    if top_k <= 0:
        top_k = 10

    _ensure_corpus_vectors()
    if not _CORPUS_TEXTS:
        return {"metadata": []}

    qv: Vector = _EMBEDDER(msg) if _EMBEDDER is not None else _embed_query_default(msg)
    if not qv:
        return {"metadata": []}

    # normalize query vector similarly to corpus vectors
    is_list_vec = False
    try:
        import numpy as _np
        if _EMBEDDER is not None and hasattr(qv, 'astype'):
            q_arr = _np.array(qv, dtype=_np.float32)
            nq = _np.linalg.norm(q_arr)
            if nq != 0:
                q_arr = (q_arr / nq)
            qv_norm = q_arr.tolist()
            is_list_vec = True
        else:
            # list/tuple path
            if isinstance(qv, (list, tuple)):
                q_list = [float(x) for x in qv]
                norm = math.sqrt(sum(x*x for x in q_list))
                if norm != 0:
                    qv_norm = [x / norm for x in q_list]
                else:
                    qv_norm = q_list
                is_list_vec = True
            else:
                # dict path (tf-idf)
                qv_norm = qv
                is_list_vec = False
    except Exception:
        # safe fallback
        if isinstance(qv, (list, tuple)):
            q_list = [float(x) for x in qv]
            norm = math.sqrt(sum(x*x for x in q_list))
            qv_norm = [x / norm for x in q_list] if norm != 0 else q_list
            is_list_vec = True
        else:
            qv_norm = qv
            is_list_vec = False

    sims: List[Tuple[int, float]] = []
    for i, dv in enumerate(_VECS):
        if is_list_vec and isinstance(dv, (list, tuple)) and len(dv) == len(qv_norm):
            # dot product (both normalized => cosine)
            raw = float(sum((a * b) for a, b in zip(dv, qv_norm)))
        else:
            # fallback to dict cosine
            raw = float(_cosine(qv_norm, dv))
        sims.append((i, raw))

    if not sims:
        return {"metadata": []}

    sims.sort(key=lambda x: x[1], reverse=True)
    # take top_k entries
    topk = sims[:top_k]
    metadata = []
    for i, raw in topk:
        item = {"text": _CORPUS_TEXTS[i], "raw": float(raw), "sem_thres": float(round(raw, 6)), "rid": int(i), "score": float(round(raw, 6))}
        # attach meta if available
        try:
            meta = _CORPUS_METAS[i] if i < len(_CORPUS_METAS) else {}
            if isinstance(meta, dict) and meta:
                item["meta"] = meta
        except Exception:
            pass
        metadata.append(item)
    return {"metadata": metadata}
