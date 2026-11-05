# bm25.py
# -*- coding: utf-8 -*-
from typing import List, Dict, Any, Optional, Iterable
import numpy as np
import re, math

_CLEAN_RE = re.compile(r"[^0-9A-Za-zÀ-Ỵà-ỵ\s]+", flags=re.UNICODE)
_WORD_RE  = re.compile(r"[0-9A-Za-zÀ-Ỵà-ỵ]+", flags=re.UNICODE)

def _normalize_text(s: str) -> str:
    if not s: return ""
    s = s.lower()
    s = _CLEAN_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _tok(s: str) -> List[str]:
    if not s: return []
    return _WORD_RE.findall(_normalize_text(s))

class BM25:
    """
    API:
        bm = BM25(docs, k1=1.5, b=0.75, stopwords=None)
        hits = bm.search(query, topk=5, min_keyword_len=2)

    Trả về mỗi hit:
        {
          "rid": int,
          "score_raw": float,       # BM25 gốc
          "score": float,           # Chuẩn hoá thận trọng: (score_raw / UB_equal) * coverage  ∈ [0..1]
          "coverage": float,        # IDF-weighted coverage ∈ [0..1]
          "matched_terms": int,     # số term duy nhất khớp
        }
    """
    def __init__(self, docs: List[str], k1: float=1.5, b: float=0.75,
                 stopwords: Optional[Iterable[str]] = None):
        self.k1, self.b = float(k1), float(b)
        self.docs_tokens: List[List[str]] = [_tok(t) for t in docs]
        if stopwords:
            sw = set(stopwords)
            self.docs_tokens = [[t for t in d if t not in sw] for d in self.docs_tokens]

        self.N = len(self.docs_tokens)
        self.avgdl = (sum(len(d) for d in self.docs_tokens) / self.N) if self.N else 0.0

        # DF/IDF (Robertson–Sparck Jones)
        self.df: Dict[str, int] = {}
        for d in self.docs_tokens:
            for t in set(d):
                self.df[t] = self.df.get(t, 0) + 1

        self.idf: Dict[str, float] = {}
        for t, df in self.df.items():
            self.idf[t] = math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))

    # ---- tính điểm BM25 cho doc i với danh sách term q_terms (unique) + tf_doc ----
    def _score_doc(self, q_terms: List[str], tf_doc: Dict[str,int], dl: int, K: float) -> float:
        s = 0.0
        for t in q_terms:
            ft = tf_doc.get(t)
            if not ft: 
                continue
            idf = self.idf.get(t, 0.0)
            s += idf * (ft * (self.k1 + 1.0)) / (ft + K)
        return s

    def search(self, query: str, topk: int = 3, min_keyword_len: int = 2) -> List[Dict[str, Any]]:
        if self.N == 0:
            return []

        # Token truy vấn (unique, giữ thứ tự) và TF của truy vấn
        q_all = [t for t in _tok(query) if len(t) >= min_keyword_len]
        if not q_all:
            return []
        q_terms = list(dict.fromkeys(q_all))
        tf_q: Dict[str, int] = {}
        for t in q_all:
            tf_q[t] = tf_q.get(t, 0) + 1

        # ---- Upper bound khi doc == query (điểm “chuẩn hoá = 1”)
        dl_equal = sum(tf_q.values())
        if self.avgdl > 0:
            K_equal = self.k1 * (1.0 - self.b + self.b * (dl_equal / self.avgdl))
        else:
            K_equal = self.k1
        UB_equal = 0.0
        for t in q_terms:
            idf = self.idf.get(t, 0.0)
            ft_q = tf_q[t]
            UB_equal += idf * (ft_q * (self.k1 + 1.0)) / (ft_q + K_equal)
        if UB_equal <= 0.0:
            UB_equal = 1e-9  # tránh chia 0

        # ---- Tính score & coverage cho từng doc
        scores = np.zeros(self.N, dtype=np.float64)
        cover  = np.zeros(self.N, dtype=np.float64)
        matched = np.zeros(self.N, dtype=np.int32)

        for i, d in enumerate(self.docs_tokens):
            dl = len(d)
            if dl == 0:
                continue

            # TF doc
            tf_doc: Dict[str, int] = {}
            for t in d:
                tf_doc[t] = tf_doc.get(t, 0) + 1

            # K của chính doc i
            K_i = self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl)) if self.avgdl > 0 else self.k1

            # raw BM25
            scores[i] = self._score_doc(q_terms, tf_doc, dl, K_i)

            # coverage theo IDF của các term có mặt ít nhất 1 lần
            matched_terms = [t for t in q_terms if t in tf_doc]
            matched[i] = len(matched_terms)
            denom = sum(self.idf.get(t, 0.0) for t in q_terms) or 1e-9
            num   = sum(self.idf.get(t, 0.0) for t in matched_terms)
            cover[i] = num / denom

        # ---- Chuẩn hoá thận trọng (giảm bão hoà 1.00 cho câu không liên quan)
        # Ý tưởng: chỉ đạt 1.0 khi (gần) khớp hoàn toàn về nội dung lẫn độ phủ từ khoá.
        # - Tỉ lệ cường độ: ratio = min(score_raw / UB_equal, 1.0)
        # - Độ phủ: kết hợp giữa coverage theo IDF (đã tính) và tỉ lệ số term truy vấn xuất hiện.
        #   coverage_terms = matched_terms / |q_terms|
        #   coverage_final = 0.5 * coverage_idf + 0.5 * coverage_terms
        # - Điểm cuối: score = ratio * coverage_final  ∈ [0..1]
        # Điều này tránh việc top-1 luôn thành 1.0 khi truy vấn rác/không liên quan.

        qlen = max(len(q_terms), 1)
        coverage_terms = matched.astype(np.float64) / float(qlen)
        coverage_final = 0.5 * cover + 0.5 * coverage_terms

        denom = max(UB_equal, 1e-9)
        ratio = np.clip(scores / denom, 0.0, 1.0)
        score_norm = ratio * coverage_final

        # ---- Chọn top-k theo score_norm (ổn định hơn min-max)
        k = max(0, int(topk))
        idx = np.argsort(-score_norm)[:k]

        out: List[Dict[str, Any]] = []
        for i in idx:
            if scores[i] <= 0:
                continue
            out.append({
                "rid": int(i),
                "score_raw": float(scores[i]),
                "score": float(score_norm[i]),      # dùng cái này để đặt ngưỡng 0.90
                "coverage": float(cover[i]),
                "matched_terms": int(matched[i]),
            })
        return out