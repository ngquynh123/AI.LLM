# services/compare_sematic_service.py
# -*- coding: utf-8 -*-
"""
Service: compare_sematic  (đánh giá NHÁNH SEMANTIC cho luồng tổng)
Input : { "metadata": [ {text, sem_thres} ], "thres": float }
Output: { "command_sematic": str, "thres": float }

Ánh xạ đúng luồng bạn yêu cầu:
- Nếu SEM_max ≥ 0.50 -> "SEMANTIC_GO_LLM" (build context + BM25 hints -> Prompt -> LLM)
- Nếu SEM_max < 0.50 -> "SEMANTIC_LOW_CONF" (không gọi LLM; nếu có BM25 thì gợi ý Top-3)
"""
from __future__ import annotations
from typing import Dict, List

SEM_GO_LLM = 0.50

def compare_semantic_service(input_data: Dict) -> Dict:
    meta: List[Dict] = (input_data or {}).get("metadata", []) or []
    base = float((input_data or {}).get("thres", 0.0))

    best = base
    for it in meta:
        try:
            best = max(best, float(it.get("sem_thres", 0.0)))
        except Exception:
            pass

    if best >= SEM_GO_LLM:
        return {"command_sematic": "SEMANTIC_GO_LLM", "thres": float(round(best, 6))}
    
    return {"command_sematic": "SEMANTIC_LOW_CONF", "thres": float(round(best, 6))}

