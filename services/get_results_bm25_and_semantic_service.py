from __future__ import annotations
from typing import Dict, Any, Callable, List, Optional
import unicodedata, re

from services import bm25_service, embed_and_semantic_service, compare_semantic_service
from services import load_data_BM25_mock, load_data_semantic_mock, preprocess_data_service
from services.preprocess_data_service import normalize_text as _norm_text


# LLM caller injection (user can set their own)
_LLM_CALLER: Optional[Callable[[str], str]] = None

def set_llm_caller(func: Callable[[str], str]) -> None:
    """Set a synchronous LLM caller: func(prompt)->str"""
    global _LLM_CALLER
    _LLM_CALLER = func


def _default_build_prompt(context_block: List[Dict[str, Any]], bm25_hints: List[Dict[str, Any]], query: str) -> str:
    parts: List[str] = []
    parts.append("Context:")
    for i, it in enumerate(context_block, start=1):
        parts.append(f"{i}. {it.get('text','')}")
    parts.append("BM25 Hints:")
    for i, it in enumerate(bm25_hints, start=1):
        parts.append(f"{i}. Q: {it.get('ques','')} A: {it.get('ans','')}")
    parts.append("User Query:")
    parts.append(query)
    parts.append("Answer:")
    return "\n".join(parts)


def run(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main pipeline entry.
    input_data: {"text": str, "top_k"?: int}
    returns dict with status and relevant payload.
    """
    text = (input_data or {}).get("text", "") or ""
    if not text.strip():
        return {"status": "empty_query"}

    top_k = int((input_data or {}).get("top_k", 5))
    if top_k <= 0:
        top_k = 5

    # preprocess the incoming text (use project's preprocess utility)
    try:
        prepped = preprocess_data_service.preprocess_text(text)
    except Exception:
        prepped = _norm_text(text)
    q = _norm_text(prepped)

    # If no semantic provider set, try to load local embedded corpus via services.load_data_semantic_mock
    try:
        if getattr(embed_and_semantic_service, "_CORPUS_PROVIDER", None) is None:
            mat, records_sem = load_data_semantic_mock.load_corpus()
            texts = [r.get("text") for r in records_sem]

            def _sem_provider_from_records():
                for t in texts:
                    yield t

            embed_and_semantic_service.set_corpus_provider(_sem_provider_from_records)
            # inject precomputed vectors into embed service so it doesn't re-embed
            try:
                vecs = [row.tolist() for row in mat]
                embed_and_semantic_service._CORPUS_TEXTS = texts
                embed_and_semantic_service._VECS = vecs
                embed_and_semantic_service._BUILT_FROM_TFIDF = True
            except Exception:
                pass
    except Exception:
        pass

    # If no BM25 provider set, try to load local data via services.load_data_BM25_mock
    try:
        if getattr(bm25_service, "_CORPUS_PROVIDER", None) is None:
            records, docs = load_data_BM25_mock.load_data_bm25()

            def _bm25_provider_from_records():
                for r in records:
                    qa = r.get("qa") or {}
                    q_ = qa.get("question") or None
                    a_ = qa.get("answer") or None
                    if q_ is not None:
                        yield q_, a_ or ""

            bm25_service.set_corpus_provider(_bm25_provider_from_records)
    except Exception:
        pass

    # 1) Semantic step first (per requested flow)
    sem_out = embed_and_semantic_service.run({"message": prepped, "top_k": top_k})
    sem_meta = sem_out.get("metadata", [])
    sem_dec = compare_semantic_service.compare_semantic_service({"metadata": sem_meta, "thres": 0.0})
    sem_cmd = sem_dec.get("command_sematic")

    # 2) BM25 retrieval (we need BM25 hints for LLM path and also to check emptiness)
    # 2) BM25 retrieval (global hints)
    bm25_out = bm25_service.run({"text": q, "top_k": top_k})
    bm25_meta = bm25_out.get("metadata", [])

    # If no BM25 results at all -> no result
    if not bm25_meta:
        return {"status": "no_bm25_result", "bm25_meta": bm25_meta}

    # derive top BM25 normalized thres (0.0 if no candidates)
    try:
        top_bm25_thres = float(bm25_meta[0].get("thres", 0.0)) if bm25_meta else 0.0
    except Exception:
        top_bm25_thres = 0.0

    # 2a) Exam-specific BM25 using qa600 subset
    exam_top_thres = 0.0
    exam_top_answer: Optional[str] = None
    try:
        records_all, _ = load_data_BM25_mock.load_data_bm25()
        qa600_records = [r for r in records_all if r.get("channel") == "qa600"]
        if qa600_records:
            from services.bm25_service import BM25, tokenize, normalize_text, _minmax_norm

            docs_tokens = [tokenize(normalize_text(((r.get("qa") or {}).get("question") or ""))) for r in qa600_records]
            idx = BM25(docs_tokens) if docs_tokens else None
            if idx and idx.N > 0:
                # get raw scores for all docs in qa600 index
                ranked_all = idx.search(q, k=idx.N)
                if ranked_all:
                    # compute UB_equal for this query on the qa600 index (same formula as in bm25_service)
                    q_tokens = tokenize(normalize_text(q))
                    if q_tokens:
                        tf_q = {}
                        for t in q_tokens:
                            tf_q[t] = tf_q.get(t, 0) + 1
                        q_terms = list(dict.fromkeys(q_tokens))
                        dl_equal = sum(tf_q.values())
                        if idx.avgdl > 0:
                            K_equal = idx.k1 * (1.0 - idx.b + idx.b * (dl_equal / idx.avgdl))
                        else:
                            K_equal = idx.k1
                        UB_equal = 0.0
                        for t in q_terms:
                            idf_t = idx.idf.get(t, 0.0)
                            ft_q = tf_q.get(t, 0)
                            if ft_q + K_equal != 0:
                                UB_equal += idf_t * (ft_q * (idx.k1 + 1.0)) / (ft_q + K_equal)
                        if UB_equal <= 0.0:
                            UB_equal = 1e-9

                        # normalize raw scores by UB_equal and clip to 1.0
                        norm_by_idx = {i: min(float(raw) / UB_equal, 1.0) for (i, raw) in ranked_all}
                        top_i, raw_top = ranked_all[0]
                        exam_top_thres = float(norm_by_idx.get(top_i, 0.0))
                        exam_top_answer = (qa600_records[top_i].get("qa") or {}).get("answer")
    except Exception:
        exam_top_thres = 0.0

    # Decision mapping requested by user (final order):
    # 1) definite_exam_answer: if exam_top_thres >= 0.9 (qa600 check)
    # 2) definite_retrieve_answer: if exam_top_thres < 0.9 AND top_bm25_thres <= 0.9 AND sem_max >= 0.55
    # 3) suggest_top3_bm25: if sem_max < 0.55 AND top_bm25_thres > 0
    # 4) no_bm25_result: if sem_max < 0.55 AND top_bm25_thres == 0
    sem_max = float(sem_dec.get("thres", 0.0))

    # 1) exam fast path
    if exam_top_thres >= 0.9:
        # sanitize sem_meta and return only top-1 sem (strip internal keys)
        def _sanitize_item(it: dict) -> dict:
            it2 = dict(it)
            for k in ("rid", "raw", "matched_terms", "score_raw", "score", "coverage"):
                if k in it2:
                    del it2[k]
            return it2

        sem_top1 = [_sanitize_item(sem_meta[0])] if sem_meta else []

        return {
            "status": "definite_exam_answer",
            "bm_result": exam_top_answer,
            "exam_thres": exam_top_thres,
            "sem_result": sem_top1,
        }

    # 2) definite_retrieve_answer (semantic confident enough)
    if exam_top_thres < 0.9 and top_bm25_thres <= 0.9 and sem_max >= 0.55:
        context_block = sem_meta[:4]
        bm25_hints = bm25_meta[:3]
        prompt = _default_build_prompt(context_block, bm25_hints, prepped)
        if _LLM_CALLER is None:
            return {
                "status": "definite_retrieve_answer",
                "bm_result": None,
                "sem_result": [{k:v for k,v in (sem_meta[0] or {}).items() if k not in ("rid","raw","matched_terms","score_raw","score","coverage")}] if sem_meta else [],
                "reason": "no_llm_configured",
                "prompt": prompt,
                "bm25_hints": bm25_hints,
                "sem_thres": sem_max,
            }
        answer = _LLM_CALLER(prompt)
        return {
            "status": "definite_retrieve_answer",
            "bm_result": answer,
            # return only top-1 sem result and sanitize bm25 hints (only first hint, strip internals)
            "sem_result": [{k: v for k, v in (sem_meta[0] or {}).items() if k not in ("rid","raw","matched_terms","score_raw","score","coverage")} ] if sem_meta else [],
            "bm25_hints": [{k: v for k, v in (bm25_hints[0] or {}).items() if k not in ("rid", "raw", "matched_terms", "score_raw", "score", "coverage")} ] if bm25_hints else [],
            "sem_thres": sem_max,
        }

    # 3) suggest_top3_bm25 (semantic low confidence but BM25 has candidates)
    if sem_max < 0.55 and top_bm25_thres > 0.0:
        # supply top-3 BM25 suggestions and top1 as bm_result
        top3 = bm25_meta[:3]
        sanitized_top3 = []
        for it in top3:
            # strip internal keys including 'rid' so callers don't receive internal ids
            sanitized_top3.append({k: v for k, v in it.items() if k not in ("rid", "raw", "matched_terms", "score_raw", "score", "coverage")})
        return {
            "status": "suggest_top3_bm25",
            "suggestions": sanitized_top3
        
        }

    # 4) no_bm25_result (semantic low confidence and BM25 empty/zero)
    if sem_max < 0.55 and top_bm25_thres == 0.0:
        return {
            "status": "no_bm25_result",
            "bm_result": None,
            "sem_result": [{k:v for k,v in (sem_meta[0] or {}).items() if k not in ("rid","raw","matched_terms","score_raw","score","coverage")} ] if sem_meta else [],
        }

    # Fallback: return recommendations (safe default)
    return {
        "status": "suggest_top3_bm25",
        "bm_result": bm25_meta[0].get("ans") if bm25_meta else None,
    "suggestions": [{k: v for k, v in it.items() if k not in ("rid", "raw", "matched_terms", "score_raw", "score", "coverage")} for it in (bm25_meta[:3] or [])],
    "sem_result": [{k:v for k,v in (sem_meta[0] or {}).items() if k not in ("rid","raw","matched_terms","score_raw","score","coverage")} ] if sem_meta else [],
        "sem_thres": sem_max,
        "bm25_thres": top_bm25_thres,
        "exam_thres": exam_top_thres,
    }


