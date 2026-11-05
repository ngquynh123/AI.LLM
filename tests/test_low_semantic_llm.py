# -*- coding: utf-8 -*-

import asyncio
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) in sys.path:
    sys.path.remove(str(TESTS_DIR))
PACKAGE_ROOT = TESTS_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from retriever import api_server


class _StaticSem:
    def __init__(self, hits):
        self._hits = hits

    def search(self, query, topk):  # pragma: no cover - simple stub
        return list(self._hits)


class _StaticBM25:
    def __init__(self, hits):
        self._hits = hits

    def search(self, query, topk, min_keyword_len):  # pragma: no cover - simple stub
        return list(self._hits)


class _SwitchingBM25:
    """Return different results for successive calls."""

    def __init__(self, first_hits, second_hits):
        self._first = first_hits
        self._second = second_hits
        self._counter = 0

    def search(self, query, topk, min_keyword_len):  # pragma: no cover - simple stub
        self._counter += 1
        if self._counter == 1:
            return list(self._first)
        return list(self._second)


def _build_minimal_args():
    args = api_server._build_default_args()
    args.llm_off = False
    args.enable_answer_preface = False
    args.enable_memory = False
    args.min_query_chars = 1
    args.min_keyword_len = 1
    args.topk = 3
    args.bm25_topn = 3
    args.ctx_topn = 3
    args.ctx_join_window = 0
    args.ctx_item_max_chars = 2000
    args.ctx_max_chars = 4000
    args.qa600_topn = 3
    args.prompt_temperature = 0.2
    args.llm_max_new = 128
    return args


def test_low_semantic_dialogue_path_uses_llm(monkeypatch):
    state_backup = dict(api_server.STATE)
    try:
        args = _build_minimal_args()
        records = [
            {
                "text": "Không được quay đầu xe tại nơi có vạch kẻ liền hai bên đường.",
                "channel": "lesson",
                "meta": {"subject": "Luật ATGT", "lesson": "5"},
            }
        ]
        dialogue_records = [
            {
                "qa": {
                    "question": "Có được quay đầu xe ở đoạn đường có vạch kẻ liền không?",
                    "answer": "Không, vạch kẻ liền báo hiệu cấm quay đầu xe.",
                },
                "text": "",
                "channel": "dialogue",
                "meta": {"subject": "Luật ATGT", "lesson": "5"},
            }
        ]
        se_hits = [{"rid": 0, "score": 0.2}]
        bm_hits = [{"rid": 0, "score": 0.15}]
        dialogue_hits = [{"rid": 0, "score": 0.12}]

        async def fake_try_llm(question, context, **_):
            assert "quay đầu xe" in context.lower()
            assert "vạch kẻ" in context.lower()
            return {
                "answer": "Không.\n- Vạch kẻ liền báo hiệu cấm quay đầu.\n- Việc quay đầu tại vị trí này gây nguy hiểm.",
                "error": None,
                "error_type": None,
                "slot_acquired": False,
                "wait_duration": 0.0,
            }

        monkeypatch.setattr(api_server, "_try_llm_generation", fake_try_llm)

        api_server.STATE.clear()
        api_server.STATE.update(
            {
                "args": args,
                "records": records,
                "dialogue_records": dialogue_records,
                "qa_records": [],
                "sem": _StaticSem(se_hits),
                "bm25_all": _StaticBM25(bm_hits),
                "bm25_dialogue": _StaticBM25(dialogue_hits),
                "bm25_qa": _StaticBM25([]),
                "llm": object(),
                "mem_chain": None,
                "conv": {},
                "query_semaphore": None,
                "query_queue_timeout": 10.0,
            }
        )

        req = api_server.QueryRequest(question="Tôi có được quay đầu xe ở vạch liền không?", session_id="pytest")
        resp = asyncio.run(api_server.query(req))
        assert resp.answer["type"] == "low-semantic-llm"
        assert "Không" in resp.answer["text"]
        assert "-" in resp.answer["text"]  # bullet enrichment
        assert resp.notice.startswith("Điểm semantic thấp")
        assert "dialogue" in resp.details
        assert resp.details["dialogue"]["context"].strip() != ""
    finally:
        api_server.STATE.clear()
        api_server.STATE.update(state_backup)


def test_low_semantic_bm25_path_uses_llm(monkeypatch):
    state_backup = dict(api_server.STATE)
    try:
        args = _build_minimal_args()
        records = [
            {
                "text": "Khi trời mưa, cần giảm tốc độ và giữ khoảng cách an toàn với xe phía trước.",
                "channel": "lesson",
                "meta": {"subject": "Kỹ thuật lái xe", "lesson": "3"},
            }
        ]
        se_hits = [{"rid": 0, "score": 0.25}]
        fallback_hits = [{"rid": 0, "score": 0.1}]

        async def fake_try_llm(question, context, **_):
            assert "trời mưa" in context.lower()
            return {
                "answer": "Nên giảm tốc độ và tăng khoảng cách an toàn khi trời mưa.",
                "error": None,
                "error_type": None,
                "slot_acquired": False,
                "wait_duration": 0.0,
            }

        monkeypatch.setattr(api_server, "_try_llm_generation", fake_try_llm)

        api_server.STATE.clear()
        api_server.STATE.update(
            {
                "args": args,
                "records": records,
                "dialogue_records": [],
                "qa_records": [],
                "sem": _StaticSem(se_hits),
                "bm25_all": _SwitchingBM25(first_hits=[], second_hits=fallback_hits),
                "bm25_dialogue": _StaticBM25([]),
                "bm25_qa": _StaticBM25([]),
                "llm": object(),
                "mem_chain": None,
                "conv": {},
                "query_semaphore": None,
                "query_queue_timeout": 10.0,
            }
        )

        req = api_server.QueryRequest(question="Trời mưa nên lái xe thế nào?", session_id="pytest")
        resp = asyncio.run(api_server.query(req))
        assert resp.answer["type"] == "low-semantic-llm"
        assert "giảm tốc độ" in resp.answer["text"].lower()
        assert resp.details["bm25"]["context"].strip() != ""
    finally:
        api_server.STATE.clear()
        api_server.STATE.update(state_backup)


def test_high_semantic_reasoning_answer_is_preserved(monkeypatch):
    state_backup = dict(api_server.STATE)
    try:
        args = _build_minimal_args()
        args.enable_answer_preface = False
        records = [
            {
                "text": (
                    "[1]\nBiển cảnh báo đường trơn yêu cầu người lái giảm tốc độ, giữ khoảng cách và quan sát mặt đường.\n"
                    "[2]\nNếu tầm nhìn hạn chế, nên bật đèn cảnh báo để xe khác nhận biết."
                ),
                "channel": "lesson",
                "meta": {"subject": "Kỹ thuật lái xe", "lesson": "4"},
            }
        ]
        se_hits = [{"rid": 0, "score": 0.82}]
        bm_hits = [{"rid": 0, "score": 0.44}]

        original_marker = api_server._marks_reasoned_sections
        seen_markers = []

        def tracking_marker(text):  # pragma: no cover - debugging aid
            seen_markers.append(text)
            return original_marker(text)

        monkeypatch.setattr(api_server, "_marks_reasoned_sections", tracking_marker)

        async def fake_async_generate_answer(llm, question, context, **_):
            assert "đường trơn" in context.lower()
            return (
                "Tóm tắt: Khi gặp biển cảnh báo đường trơn, hãy giảm tốc và giữ khoảng cách an toàn (tham khảo [1]).\n"
                "Suy luận hợp lý: Bật đèn cảnh báo chống trượt giúp xe phía sau nhận biết (tham khảo [2])."
            )

        monkeypatch.setattr(api_server, "async_generate_answer", fake_async_generate_answer)

        api_server.STATE.clear()
        api_server.STATE.update(
            {
                "args": args,
                "records": records,
                "dialogue_records": [],
                "qa_records": [],
                "sem": _StaticSem(se_hits),
                "bm25_all": _StaticBM25(bm_hits),
                "bm25_dialogue": _StaticBM25([]),
                "bm25_qa": _StaticBM25([]),
                "llm": object(),
                "mem_chain": None,
                "conv": {},
                "query_semaphore": None,
                "query_queue_timeout": 10.0,
            }
        )

        req = api_server.QueryRequest(
            question="Biển cảnh báo đường trơn nhắc người lái cần chú ý điều gì?",
            session_id="pytest",
        )
        resp = asyncio.run(api_server.query(req))
        assert resp.answer["type"] == "semantic"
        assert "Suy luận hợp lý" in resp.answer["text"]
        assert "Chưa đủ căn cứ" not in resp.answer["text"]
        assert any(isinstance(t, str) and "suy luận hợp lý" in t.lower() for t in seen_markers), seen_markers
    finally:
        api_server.STATE.clear()
        api_server.STATE.update(state_backup)


def test_numeric_gap_injects_reasoning_label(monkeypatch):
    state_backup = dict(api_server.STATE)
    try:
        args = _build_minimal_args()
        args.enable_answer_preface = False
        records = [
            {
                "text": (
                    "[1]\nQuy định về tốc độ tối đa giúp đảm bảo an toàn và tránh va chạm."\
                    "\n[2]\nTrong khu dân cư chỉ nhấn mạnh việc giảm tốc độ, không nêu con số cố định."
                ),
                "channel": "lesson",
                "meta": {"subject": "Luật GT", "lesson": "3"},
            }
        ]
        se_hits = [{"rid": 0, "score": 0.73}]
        bm_hits = [{"rid": 0, "score": 0.4}]

        async def fake_async_generate_answer(llm, question, context, **_):
            assert "tốc độ" in question.lower()
            return "Tốc độ tối đa trong khu dân cư thường khoảng 50 km/h để giảm nguy cơ tai nạn."

        monkeypatch.setattr(api_server, "async_generate_answer", fake_async_generate_answer)

        api_server.STATE.clear()
        api_server.STATE.update(
            {
                "args": args,
                "records": records,
                "dialogue_records": [],
                "qa_records": [],
                "sem": _StaticSem(se_hits),
                "bm25_all": _StaticBM25(bm_hits),
                "bm25_dialogue": _StaticBM25([]),
                "bm25_qa": _StaticBM25([]),
                "llm": object(),
                "mem_chain": None,
                "conv": {},
                "query_semaphore": None,
                "query_queue_timeout": 10.0,
            }
        )

        req = api_server.QueryRequest(
            question="Tốc độ tối đa cho phép trong khu dân cư thường là bao nhiêu km/h?",
            session_id="pytest",
        )
        resp = asyncio.run(api_server.query(req))
        text = resp.answer["text"]
        assert resp.answer["type"] == "semantic"
        assert "Suy luận hợp lý" in text
        assert "50 km/h" in text
        assert "Ghi chú" in text
        assert "Chưa đủ căn cứ" not in text
    finally:
        api_server.STATE.clear()
        api_server.STATE.update(state_backup)
