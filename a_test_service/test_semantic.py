# tests/test_semantic.py
# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.embed_and_semantic_service import set_corpus_provider, run
from a_test_service.providers_demo import semantic_corpus_provider

def main():
    # 1) Gán corpus text
    set_corpus_provider(semantic_corpus_provider)

    # 2) Gọi service
    message = "biển cấm viền đỏ nền trắng"
    out = run({"message": message, "top_k": 5})

    # 3) In kết quả
    print("=== SEMANTIC RESULT ===")
    print("Message:", message)
    for i, item in enumerate(out.get("metadata", []), start=1):
        print(f"{i:02d}. sem_thres={item['sem_thres']:.4f} | text={item['text']}")

if __name__ == "__main__":
    main()
