# tests/test_pipeline_decision.py
# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.bm25_service import set_corpus_provider as set_bm25_provider
from services.embed_and_semantic_service import set_corpus_provider as set_sem_provider
from services.get_results_bm25_and_semantic_service import run as pipeline_run, set_llm_caller
from services import load_data_BM25_mock, load_data_semantic_mock


def pipeline(message: str):
    # delegate to the new pipeline service
    return pipeline_run({"text": message, "top_k": 5})

def main():
    # Load local BM25/semantic data and register providers
    try:
        records_bm25, docs_bm25 = load_data_BM25_mock.load_data_bm25()
    except Exception:
        records_bm25, docs_bm25 = [], []

    def _bm25_provider_from_records():
        for r in records_bm25:
            qa = r.get("qa") or {}
            q = qa.get("question")
            a = qa.get("answer")
            if q:
                yield q, a or ""

    set_bm25_provider(_bm25_provider_from_records)

    try:
        mat, records_sem = load_data_semantic_mock.load_corpus()
    except Exception:
        records_sem = []

    def _sem_provider_from_records():
        for r in records_sem:
            t = r.get("text")
            if t:
                yield t

    set_sem_provider(_sem_provider_from_records)

    # install a dummy LLM caller so SEMANTIC_GO_LLM path returns an answer
    def _dummy_llm(prompt: str) -> str:
        return "[DUMMY LLM ANSWER] Based on provided context."

    set_llm_caller(_dummy_llm)

    tests = [
        "khi hệ thống phanh bị bó hoặc sử dụng liên tục trên quãng đường dài, hậu quả có thể là gì",   # kỳ vọng match mạnh
        "ô tô được phân loại theo gì",          # semantic đi LLM
        "câu hỏi ngoài phạm vi dữ liệu",    # low conf -> gợi ý BM25
    ]
    for q in tests:
        print("\n==============================")
        print("Query:", q)
        result = pipeline(q)
        print("Result:", result)

if __name__ == "__main__":
    main()
