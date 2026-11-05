from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from services.bm25_service import set_corpus_provider, run
from a_test_service.providers_demo import bm25_provider

def main():
    # 1) Gán nguồn dữ liệu cho BM25
    set_corpus_provider(bm25_provider)

    # 2) Gọi service
    query = "biển báo cấm nền trắng viền đỏ"
    out = run({"text": query, "top_k": 5})

    # 3) In kết quả
    print("=== BM25 RESULT ===")
    print("Query:", query)
    for i, item in enumerate(out.get("metadata", []), start=1):
        print(f"{i:02d}. thres={item['thres']:.4f} | ques={item['ques']} | ans={item['ans']}")

if __name__ == "__main__":
    main()
