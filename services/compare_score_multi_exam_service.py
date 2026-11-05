# services/compare_score_multi_exam_service.py
# -*- coding: utf-8 -*-
"""
Service: compare_score_multi_exam  (đánh giá NHÁNH BM25)
Input : { "thres": float, "metadata": [ {ques, ans, thres} ] }
Output: { "command_bm25": str, "thres": float }

Luồng quyết định theo yêu cầu:
- Nếu không có kết quả BM25 -> command_bm25 = "BM25_NONE"
- Nếu top1.thres >= 0.90     -> "BM25_FAST_PATH_DEFINITE_ANSWER" (trả lời trực tiếp)
- Còn lại                     -> "BM25_HAS_CANDIDATES" (đi các bước SEM/RRF sau)
"""
from __future__ import annotations
from typing import Dict, List

FAST_THRES = 0.90

def compare_score_multi_exam_service(input_data: Dict) -> Dict:
    meta: List[Dict] = (input_data or {}).get("metadata", []) or []
    base_thres = float((input_data or {}).get("thres", 0.0))

    if not meta:
        return {"command_bm25": "BM25_NONE", "thres": 0.0}

    # top1 theo thứ tự đã sort sẵn ở service bm25
    top1 = meta[0]
    best = max(base_thres, float(top1.get("thres", 0.0)))

    if best >= FAST_THRES:
        return {"command_bm25": "BM25_FAST_PATH_DEFINITE_ANSWER", "thres": float(round(best, 6))}
    return {"command_bm25": "BM25_HAS_CANDIDATES", "thres": float(round(best, 6))}

