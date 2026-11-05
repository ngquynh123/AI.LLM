
from __future__ import annotations
from typing import List, Tuple, Dict, Callable, Iterable, Optional
from collections import Counter
import math, re, unicodedata

# =======================
# Token hoá & normalize
# =======================
_TOKEN = re.compile(r"[A-Za-zÀ-ỹ0-9_]+")

def normalize_text(text: str) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFC", text)
    t = t.strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t

def tokenize(text: str) -> List[str]:
    return [w for w in _TOKEN.findall(text)]


class BM25:
    def __init__(self, docs: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = docs
        self.N = len(docs)
        self.avgdl = sum(len(d) for d in docs) / max(1, self.N)
        self.tf = [Counter(d) for d in docs]
        df = Counter()
        for c in self.tf:
            df.update(c.keys())
        # idf theo dạng log(1 + (N - df + 0.5)/(df + 0.5))
        self.idf = {w: math.log(1 + (self.N - dfw + 0.5) / (dfw + 0.5)) for w, dfw in df.items()}

    def _score_query_doc(self, q_tokens: List[str], i: int) -> float:
        c = self.tf[i]
        dl = sum(c.values())
        s = 0.0
        for w in q_tokens:
            if w not in c: 
                continue
            idf = self.idf.get(w)
            if idf is None:
                continue
            tf = c[w]
            denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            s += idf * (tf * (self.k1 + 1)) / denom
        return s

    def search(self, query_text: str, k: int = 10) -> List[Tuple[int, float]]:
        q = tokenize(normalize_text(query_text))
        if not q or self.N == 0:
            return []
        scores = [(i, self._score_query_doc(q, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

# Provider: callable trả về Iterable[Tuple[ques, ans]]
_CORPUS_PROVIDER: Optional[Callable[[], Iterable[Tuple[str, str]]]] = None

# Bộ nhớ tạm index
_INDEX: Optional[BM25] = None
_CORPUS_QA: List[Tuple[str, str]] = []
_INDEX_BUILT: bool = False

def set_corpus_provider(provider: Callable[[], Iterable[Tuple[str, str]]]) -> None:
    """
    Đăng ký nguồn dữ liệu (ques, ans).
    - Gọi hàm này một lần ở bootstrap của ứng dụng.
    - Lần đầu run() sẽ tự build index từ provider.
    """
    global _CORPUS_PROVIDER, _INDEX, _CORPUS_QA, _INDEX_BUILT
    _CORPUS_PROVIDER = provider
    _INDEX = None
    _CORPUS_QA = []
    _INDEX_BUILT = False

def _ensure_index() -> None:
    """
    Build index lần đầu từ provider đã đăng ký.
    Không có provider -> để trống (không lỗi để service vẫn an toàn khi gọi).
    """
    global _INDEX, _CORPUS_QA, _INDEX_BUILT
    if _INDEX_BUILT:
        return
    _INDEX_BUILT = True

    if _CORPUS_PROVIDER is None:
        # Không có provider => không build (run() sẽ trả metadata rỗng)
        _INDEX = BM25(docs=[])
        _CORPUS_QA = []
        return

    corpus: List[Tuple[str, str]] = []
    docs_tokens: List[List[str]] = []
    for ques, ans in _CORPUS_PROVIDER():
        corpus.append((ques, ans))
        docs_tokens.append(tokenize(normalize_text(ques or "")))
    _CORPUS_QA = corpus
    _INDEX = BM25(docs_tokens)

# Normalize score 
def _minmax_norm(scores: List[float]) -> List[float]:
    if not scores:
        return []
    mn, mx = min(scores), max(scores)
    if mx - mn == 0:
        # If all scores are identical:
        # - if they are all zero -> treat as no-match -> normalized to 0.0
        # - otherwise (all equal non-zero) keep them as 1.0 (relative equality)
        return [0.0 * 1.0 if mx == 0.0 else 1.0] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]

def run(input_data: Dict) -> Dict:
    """
    input_data:
      - text: str (bắt buộc)
      - top_k?: int (mặc định 10)

    return:
      { "metadata": [ { "ques": str, "ans": str, "thres": float } ] }
    """
    text = (input_data or {}).get("text", "") or ""
    if not text.strip():
        return {"metadata": []}

    top_k = int((input_data or {}).get("top_k", 10))
    if top_k <= 0:
        top_k = 10

    _ensure_index()
    if _INDEX is None or _INDEX.N == 0:
        # Chưa có provider hoặc corpus rỗng
        return {"metadata": []}

    # compute scores across the entire corpus
    ranked_all = _INDEX.search(text, k=_INDEX.N)
    if not ranked_all:
        return {"metadata": []}

    # Build UB_equal normalization following the retriever/bm25.py approach
    # Derive query tokens and term frequencies
    q_tokens = tokenize(normalize_text(text))
    if not q_tokens:
        return {"metadata": []}
    tf_q = {}
    for t in q_tokens:
        tf_q[t] = tf_q.get(t, 0) + 1
    # unique query terms preserving order
    q_terms = list(dict.fromkeys(q_tokens))
    dl_equal = sum(tf_q.values())
    if _INDEX.avgdl > 0:
        K_equal = _INDEX.k1 * (1.0 - _INDEX.b + _INDEX.b * (dl_equal / _INDEX.avgdl))
    else:
        K_equal = _INDEX.k1

    UB_equal = 0.0
    for t in q_terms:
        idf_t = _INDEX.idf.get(t, 0.0)
        ft_q = tf_q.get(t, 0)
        UB_equal += idf_t * (ft_q * (_INDEX.k1 + 1.0)) / (ft_q + K_equal) if ft_q + K_equal != 0 else 0.0
    if UB_equal <= 0.0:
        UB_equal = 1e-9

    # raw scores list (aligned with ranked_all order)
    raw_scores_all = [s for _, s in ranked_all]
    # normalize by UB_equal and clip to 1.0 (doc == query => 1.0)
    norm_by_idx = {idx: min(float(raw) / UB_equal, 1.0) for (idx, raw) in ranked_all}

    # take top_k from the already-sorted ranked_all
    topk = ranked_all[:top_k]
    out = []
    # prepare q_terms and denom for coverage calculation
    q_tokens = tokenize(normalize_text(text))
    tf_q = {}
    for t in q_tokens:
        tf_q[t] = tf_q.get(t, 0) + 1
    q_terms = list(dict.fromkeys(q_tokens))
    denom = sum(_INDEX.idf.get(t, 0.0) for t in q_terms) or 1e-9

    for i, raw in topk:
        ques, ans = _CORPUS_QA[i]
        z = norm_by_idx.get(i, 0.0)
        # compute coverage and matched_terms using index internals
        try:
            tf_doc = _INDEX.tf[i]
            matched_terms = sum(1 for t in q_terms if t in tf_doc)
            num = sum(_INDEX.idf.get(t, 0.0) for t in q_terms if t in tf_doc)
            coverage = float(num) / float(denom) if denom != 0 else 0.0
        except Exception:
            matched_terms = 0
            coverage = 0.0

        out.append({
            # legacy keys used elsewhere in pipeline
            "ques": ques,
            "ans": ans,
            "raw": float(raw),
            "thres": float(round(z, 6)),
            # retriever-compatible keys
            "rid": int(i),
            "score_raw": float(raw),
            "score": float(round(z, 6)),
            "coverage": float(coverage),
            "matched_terms": int(matched_terms),
        })
    return {"metadata": out}
