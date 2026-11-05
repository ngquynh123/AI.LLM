# -*- coding: utf-8 -*-
"""
FastAPI wrapper for the retriever CLI logic in main.py.
- Keeps logic, flow, thresholds, providers 100% identical
- Only changes IO: from terminal prints to JSON API

Endpoints
- GET /health -> {status: ok}
- POST /query {question: str, session_id?: str} -> JSON with the same sections as terminal output

Run (example):
  uvicorn retriever.api_server:app --host 0.0.0.0 --port 8000
"""
import asyncio
import os
import json
import math
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Literal, Optional, Set

from fastapi import FastAPI, HTTPException, Query
import logging
from pydantic import BaseModel, Field

# Import the same building blocks as main.py
from .load_data import load_corpus
from .model_semantic import SemanticSearcher
from .bm25 import BM25
from .model_llm import build_llm_from_args, generate_answer
from .chat_memory import build_memory_chain_hf, resolve_after_preprocess, update_focus_after_sem
from .preprocess import clean_user_input
from services.results_final_service import _build_source_from_meta  # reuse robust source builder
from tools import async_batch_runner

# ========== Compatibility patches (align with main copy.py) ==========
# - Patch torch.nn.MultiheadAttention for older torch builds lacking dtype param
# - Provide a dummy flash_attn_triton module if not available
def _patch_multihead_attention():
    try:
        import torch.nn as nn, inspect
        if 'dtype' in inspect.signature(nn.MultiheadAttention.__init__).parameters:
            return
        if not hasattr(nn.MultiheadAttention, "_original_init"):
            nn.MultiheadAttention._original_init = nn.MultiheadAttention.__init__
        def patched_init(self, *args, **kwargs):
            for k in ["dtype", "bias", "add_bias_kv", "add_zero_attn"]:
                if k in kwargs:
                    kwargs.pop(k, None)
            return nn.MultiheadAttention._original_init(self, *args, **kwargs)
        nn.MultiheadAttention.__init__ = patched_init
    except Exception:
        pass

def _patch_flash_attn_triton():
    try:
        import flash_attn_triton  # noqa: F401
    except Exception:
        import types as _types, sys as _sys
        m = _types.ModuleType("flash_attn_triton")
        m.HAS_FLASH_ATTN = False
        m.available = lambda: False
        _sys.modules["flash_attn_triton"] = m

_patch_multihead_attention()
_patch_flash_attn_triton()

# Default to offline HF by default (can be overridden via env)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

BASE_DIR = Path(__file__).resolve().parent.parent

# ========= Helpers copied from main.py (to avoid module import issues) =========
import re, unicodedata, shutil, textwrap
from typing import Tuple

def _term_width(default: int = 100) -> int:
    try:
        import shutil as _sh
        return max(60, _sh.get_terminal_size((default, 20)).columns)
    except Exception:
        return default

def _box(title: str, body: str, width: int = None) -> str:
    width = width or _term_width()
    line = "═" * (width - 2)
    top, bot = f"╔{line}╗", f"╚{line}╝"
    head, sep = f"║ {title.strip()}".ljust(width - 1) + "║", "╟" + ("─" * (width - 2)) + "╢"
    rows = []
    for paragraph in (body or "").split("\n"):
        if not paragraph.strip():
            rows.append("║".ljust(width - 1) + "║"); continue
        import textwrap as _tw
        for w in (_tw.wrap(paragraph, width=width - 4) or [""]):
            rows.append(("║ " + w).ljust(width - 1) + "║")
    return "\n".join([top, head, sep] + rows + [bot])

_DIRECT_ANSWER_PATTERNS = [
    r"\bđáp án\s*[:\-]",
    r"\btrả lời\s*[:\-]",
    r"\bkết luận\s*[:\-]",
    r"\bchọn đáp án\b",
    r"\bđúng là\b",
    r"\blà\s*[:\-]\s",
    r"\bđịnh nghĩa\s*[:\-]",
]


def _record_has_direct_answer(rec: Dict[str, Any]) -> bool:
    qa = rec.get("qa") or {}
    ans = qa.get("answer") or rec.get("answer")
    if isinstance(ans, str) and ans.strip():
        return True
    txt = _get_text(rec)
    if not isinstance(txt, str) or not txt:
        return False
    for pat in _DIRECT_ANSWER_PATTERNS:
        try:
            if re.search(pat, txt, flags=re.IGNORECASE):
                return True
        except re.error:
            continue
    return False


def clean_text(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFC", s).lower()
    s = re.sub(r"[^a-z0-9\u00c0-\u024f\u1e00-\u1eff\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ===== Small-talk / identity intent detection =====
_HP_AI_REPLY = "Xin chào. Tôi là HP_AI, tôi có thể giúp gì cho bạn?"

_GREET_PAT = re.compile(
    r"\b(xin chao|xin chào|chao|chào|hello|hi|hey|alo|yo)\b",
    re.IGNORECASE,
)
_IDENT_PAT = re.compile(
    r"\b(ban la ai|bạn là ai|ban la gi|bạn là gì|may la ai|mày là ai|cau la ai|cậu là ai|ai vay|ai vậy|who are you|what are you|introduce yourself|gioi thieu|giới thiệu)\b",
    re.IGNORECASE,
)

def _is_smalltalk_identity(q: str) -> bool:
    qn = clean_text(q or "")
    if not qn:
        return False
    return bool(_GREET_PAT.search(qn) or _IDENT_PAT.search(qn))


# ----- Load-test JSONL helpers -----
def _resolve_workspace_path(path: str) -> Path:
    candidate = Path(path or "").expanduser()
    if not candidate.is_absolute():
        candidate = BASE_DIR / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(BASE_DIR)
    except ValueError:
        raise HTTPException(status_code=400, detail="Đường dẫn phải nằm trong thư mục ứng dụng")
    return resolved


def _percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    if pct <= 0:
        return float(min(values))
    if pct >= 1:
        return float(max(values))
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(ordered[int(k)])
    return float(ordered[f] * (c - k) + ordered[c] * (k - f))


def _init_load_summary() -> Dict[str, Any]:
    return {
        "statuses": Counter(),
        "elapsed": [],
        "busy": 0,
        "errors": 0,
    }


def _update_load_summary(state: Dict[str, Any], record: Dict[str, Any]) -> None:
    status = record.get("status")
    if status is None:
        state["errors"] += 1
    else:
        state["statuses"][status] += 1

    elapsed = record.get("elapsed_ms")
    if isinstance(elapsed, (int, float)):
        state["elapsed"].append(float(elapsed))

    busy_detected = False
    body = record.get("body")
    if isinstance(body, dict):
        answer = body.get("answer")
        if isinstance(answer, dict) and answer.get("type") == "busy":
            busy_detected = True
        elif isinstance(body.get("details"), dict) and body["details"].get("reason") == "llm_queue_timeout":
            busy_detected = True
    if busy_detected:
        state["busy"] += 1


def _finalise_load_summary(state: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not state:
        return None
    elapsed = state.get("elapsed") or []
    latency = None
    if elapsed:
        latency = {
            "p50_ms": round(_percentile(elapsed, 0.5), 2) if elapsed else None,
            "p90_ms": round(_percentile(elapsed, 0.9), 2) if elapsed else None,
            "max_ms": round(max(elapsed), 2),
        }
    return {
        "status_counts": dict(state.get("statuses", {})),
        "busy_responses": state.get("busy", 0),
        "errors": state.get("errors", 0),
        "latency": latency,
    }


def _load_jsonl_records(
    path: str,
    *,
    offset: int = 0,
    limit: Optional[int] = 50,
    include_summary: bool = False,
) -> Dict[str, Any]:
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset phải >= 0")
    if limit is not None and limit < 0:
        raise HTTPException(status_code=400, detail="limit phải >= 0 hoặc None")

    resolved_path = _resolve_workspace_path(path)
    if not resolved_path.exists():
        raise HTTPException(status_code=404, detail=f"Không tìm thấy file: {path}")

    items: List[Dict[str, Any]] = []
    total = 0
    summary_state = _init_load_summary() if include_summary else None

    with resolved_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if summary_state:
                    summary_state["errors"] += 1
                continue
            total += 1
            if summary_state is not None:
                _update_load_summary(summary_state, record)
            if total <= offset:
                continue
            if limit is not None and len(items) >= limit:
                continue
            items.append(record)

    summary = _finalise_load_summary(summary_state)
    return {
        "path": str(resolved_path),
        "total": total,
        "items": items,
        "summary": summary,
    }


def _build_runner_namespace(
    req: Any,
    questions_path: Path,
    save_path: Optional[Path],
) -> SimpleNamespace:
    return SimpleNamespace(
        url=req.url,
        questions=str(questions_path),
        total=req.total,
        concurrency=req.concurrency,
        session_id=req.session_id,
        http_timeout=req.http_timeout,
        max_retries=req.max_retries,
        dry_run=req.dry_run,
        save_responses=str(save_path) if save_path else None,
    )


def _write_jsonl(path: Path, results: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    async_batch_runner.save_jsonl(path, results)

# ===== Answer preface helper (adds a short, polite intro referencing the question) =====
_PREFACE_LEAD_PAT = re.compile(r"^(vâng|dạ|xin chào|chào|hello|hi)\b", re.IGNORECASE)

def _most_common_line(text: str) -> Optional[str]:
    if not isinstance(text, str) or not text.strip():
        return None
    lines = [ln.strip() for ln in text.splitlines() if ln and ln.strip()]
    if not lines:
        return None
    from collections import Counter
    cnt = Counter(lines)
    most, freq = cnt.most_common(1)[0]
    total = sum(cnt.values())
    # Consider it a repeated title if appears >=3 times and accounts for >=60% of lines
    if freq >= 3 and freq / max(1, total) >= 0.6:
        return most
    # If overall lines are many and unique lines are very few, also treat as repetition
    if total >= 5 and len(cnt) <= 2 and freq >= 3:
        return most
    return None

def _should_disable_preface(orig_question: str) -> bool:
    mc = _most_common_line(orig_question or "")
    return bool(mc)

def _normalize_simple(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" .,:;!?…")
    return s.lower()

def _remove_question_echo_lines(answer_text: str, orig_question: str) -> str:
    """Remove lines in the answer that echo the original question/title (or its truncated form)."""
    if not isinstance(answer_text, str) or not answer_text.strip():
        return answer_text or ""
    topic = _most_common_line(orig_question or "") or (orig_question or "").strip()
    topic_norm = _normalize_simple(topic)
    if not topic_norm:
        return answer_text
    lines = [ln for ln in (answer_text or "").splitlines()]
    out = []
    for i, ln in enumerate(lines):
        chk = _normalize_simple(ln)
        # If line equals or starts with the topic (allow ellipsis), drop it
        if chk == topic_norm:
            continue
        if topic_norm and (chk.startswith(topic_norm) or topic_norm.startswith(chk)):
            # also drop short ellipsis variations
            continue
        out.append(ln)
    return "\n".join(out).strip()

# ===== Offensive/abusive language detection (Vietnamese + common English profanity) =====
_VN_WORD_BOUND = r"[^A-Za-zÀ-ỹĐđ]"
_OFFENSIVE_PATTERNS = [
    # Insults
    rf"(^|{_VN_WORD_BOUND})ngu({_VN_WORD_BOUND}|$)",
    rf"(^|{_VN_WORD_BOUND})đần({_VN_WORD_BOUND}|$)",
    rf"(^|{_VN_WORD_BOUND})dốt({_VN_WORD_BOUND}|$)",
    rf"(^|{_VN_WORD_BOUND})óc\s*chó({_VN_WORD_BOUND}|$)",
    rf"(^|{_VN_WORD_BOUND})mất\s*dạy({_VN_WORD_BOUND}|$)",
    rf"(^|{_VN_WORD_BOUND})vô\s*học({_VN_WORD_BOUND}|$)",
    rf"(^|{_VN_WORD_BOUND})súc\s*vật({_VN_WORD_BOUND}|$)",
    rf"(^|{_VN_WORD_BOUND})cặn\s*bã({_VN_WORD_BOUND}|$)",
    rf"(^|{_VN_WORD_BOUND})khốn\s*nạn({_VN_WORD_BOUND}|$)",
    rf"(^|{_VN_WORD_BOUND})đểu({_VN_WORD_BOUND}|$)",
    rf"(^|{_VN_WORD_BOUND})bẩn\s*thỉu({_VN_WORD_BOUND}|$)",
    # Profanity (Vietnamese)
    rf"(^|{_VN_WORD_BOUND})địt({_VN_WORD_BOUND}|$)",
    rf"(^|{_VN_WORD_BOUND})đụ({_VN_WORD_BOUND}|$)",
    rf"(^|{_VN_WORD_BOUND})đéo({_VN_WORD_BOUND}|$)",
    rf"(^|{_VN_WORD_BOUND})đmm?({_VN_WORD_BOUND}|$)",
    rf"(^|{_VN_WORD_BOUND})dm({_VN_WORD_BOUND}|$)",
    rf"(^|{_VN_WORD_BOUND})vcl|vkl({_VN_WORD_BOUND}|$)",
    rf"(^|{_VN_WORD_BOUND})lồn({_VN_WORD_BOUND}|$)",
    rf"(^|{_VN_WORD_BOUND})cặc({_VN_WORD_BOUND}|$)",
    rf"(^|{_VN_WORD_BOUND})buồi({_VN_WORD_BOUND}|$)",
    # Profanity (English)
    rf"(^|{_VN_WORD_BOUND})fuck(s|ing)?({_VN_WORD_BOUND}|$)",
    rf"(^|{_VN_WORD_BOUND})shit({_VN_WORD_BOUND}|$)",
    rf"(^|{_VN_WORD_BOUND})bitch({_VN_WORD_BOUND}|$)",
    rf"(^|{_VN_WORD_BOUND})asshole({_VN_WORD_BOUND}|$)",
    rf"(^|{_VN_WORD_BOUND})motherf(\*|u)cker({_VN_WORD_BOUND}|$)",
]

def _is_offensive_question(q: str) -> bool:
    if not isinstance(q, str) or not q.strip():
        return False
    s = q.lower()
    for pat in _OFFENSIVE_PATTERNS:
        try:
            if re.search(pat, s, flags=re.IGNORECASE):
                return True
        except Exception:
            continue
    return False

def _build_answer_preface(orig_question: str) -> str:
    q = (orig_question or "").strip()
    if not q:
        return "Vâng, tôi có thể giúp gì cho bạn."
    topic = q.rstrip("?.! ")
    if len(topic) > 140:
        topic = topic[:140].rstrip() + "…"
    return f"Vâng, bạn muốn hỏi về {topic}."

def _maybe_with_preface(text: str, orig_question: str, enabled: bool) -> str:
    if not enabled:
        return (text or "").strip()
    body = (text or "").strip()
    if not body:
        # Keep a minimal preface to avoid empty answers, caller may override with extracted content later
        pre = _build_answer_preface(orig_question)
        return pre
    if _PREFACE_LEAD_PAT.match(body):
        return body
    pre = _build_answer_preface(orig_question)
    return (pre + "\n" + body).strip()

def _extract_q_and_a(r: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    qa = r.get("qa") or {}
    q = qa.get("question") or r.get("question")
    a = qa.get("answer")   or r.get("answer")
    q = q.strip() if isinstance(q, str) and q.strip() else None
    a = a.strip() if isinstance(a, str) and a.strip() else None
    return q, a


def _normalize_bm25_score(score: Any) -> float:
    try:
        val = float(score)
    except (TypeError, ValueError):
        return 0.0
    if val <= 0:
        return 0.0
    # BM25 scores in this corpus typically range 0-40; compress to [0,1]
    return min(1.0, val / 40.0)


_NUMERIC_FOCUS_PAT = re.compile(r"(bao nhiêu|mấy|tốc độ|max|giới hạn|số|khoảng cách|chỉ số|hệ số|%|km/h)", re.IGNORECASE)


_DEF_QUESTION_PAT = re.compile(r"(là gì|định nghĩa|ý nghĩa|khái niệm|mô tả|ký hiệu|gọi là)\b", re.IGNORECASE)


def _question_is_definition(question: Optional[str]) -> bool:
    if not isinstance(question, str) or not question.strip():
        return False
    return bool(_DEF_QUESTION_PAT.search(question))


def _question_focus_features(question: Optional[str]) -> Dict[str, Any]:
    q = (question or "").strip()
    numbers = re.findall(r"\d+(?:[\.,]\d+)?", q)
    has_numeric_kw = bool(_NUMERIC_FOCUS_PAT.search(q))
    return {
        "numbers": numbers,
        "numeric_focus": bool(numbers) or has_numeric_kw,
    }


def _record_matches_numbers(rec: Dict[str, Any], numbers: List[str]) -> bool:
    if not numbers:
        return False
    txt = _get_text(rec)
    if not txt:
        return False
    normalized = re.sub(r"\s+", "", txt)
    numeric_only = re.sub(r"[^0-9\.,]", "", txt)
    for num in numbers:
        raw = (num or "").replace(" ", "")
        if not raw:
            continue
        if raw in normalized:
            return True
        alt = raw.replace(",", ".")
        if alt and alt in numeric_only:
            return True
    return False


_DEF_PAT = re.compile(r"\b(là gì|định nghĩa|ý nghĩa|ký hiệu|đáp án)\b", re.IGNORECASE)


def sort_by_definition_first(records: List[Dict[str, Any]], hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def score(h: Dict[str, Any]) -> int:
        try:
            rid = h["rid"]
            txt = (records[rid].get("text") or "").lower()
            return 1 if _DEF_PAT.search(txt) else 0
        except Exception:
            return 0
    return sorted(hits, key=score, reverse=True)


def dedup_by_source(records: List[Dict[str, Any]], hits: List[Dict[str, Any]], max_per_source: int = 1) -> List[Dict[str, Any]]:
    seen: Dict[str, int] = {}
    kept: List[Dict[str, Any]] = []
    for h in hits:
        rid = h.get("rid")
        if rid is None or rid < 0 or rid >= len(records):
            continue
        rec = records[rid]
        key = f"{rec.get('channel','?')}::{rec.get('file','?')}"
        if seen.get(key, 0) >= max_per_source:
            continue
        seen[key] = seen.get(key, 0) + 1
        kept.append(h)
    return kept


def join_adjacent_chunks(records: List[Dict[str, Any]], hits: List[Dict[str, Any]], window: int = 1, max_chars: int = 1200) -> List[str]:
    out: List[str] = []
    taken: set[int] = set()
    for h in hits:
        rid = h.get("rid")
        if rid is None or rid in taken or rid < 0 or rid >= len(records):
            continue
        rec = records[rid]
        file_id = rec.get("file")
        pieces = [rec.get("text") or ""]
        for off in range(1, window + 1):
            for rrid in (rid - off, rid + off):
                if 0 <= rrid < len(records) and rrid not in taken:
                    neighbour = records[rrid]
                    if neighbour.get("file") == file_id:
                        pieces.append(neighbour.get("text") or "")
                        taken.add(rrid)
        merged = " ".join([p.strip() for p in pieces if p and p.strip()])
        slice_txt = merged[:max_chars] + ("…" if len(merged) > max_chars else "")
        out.append(slice_txt)
        taken.add(rid)
    return out


def _build_context_block_from_hits(ctx_hits: List[str], topn_ctx: int, max_chars_per_item: int, max_total_chars: int) -> str:
    blocks: List[str] = []
    total = 0
    for idx, text in enumerate(ctx_hits[:topn_ctx], 1):
        head = f"[{idx}]"
        body = (text or "").replace("\r", " ").replace("\t", " ").strip()
        if len(body) > max_chars_per_item:
            body = body[:max_chars_per_item] + "…"
        block = head + "\n" + body
        if total + len(block) > max_total_chars:
            break
        blocks.append(block)
        total += len(block)
    return "\n\n".join(blocks)


def _build_bm25_hints_block(records: List[Dict[str, Any]], bm25_hits: List[Dict[str, Any]], max_chars_per_item: int = 160) -> str:
    lines: List[str] = []
    for i, h in enumerate(bm25_hits, 1):
        rid = h.get("rid")
        if rid is None or rid < 0 or rid >= len(records):
            continue
        rec = records[rid]
        q, a = _extract_q_and_a(rec)
        txt = (q or a or _get_text(rec) or "").replace("\n", " ")
        if len(txt) > max_chars_per_item:
            txt = txt[:max_chars_per_item] + "…"
        lines.append(f"[B{i}] {txt}")
    return "\n".join(lines)


def _build_bm25_only_block(records: List[Dict[str, Any]], bm25_hits: List[Dict[str, Any]], topn: int = 3, max_chars: int = 700) -> str:
    blocks: List[str] = []
    for i, h in enumerate(bm25_hits[:topn], 1):
        rid = h.get("rid")
        if rid is None or rid < 0 or rid >= len(records):
            continue
        rec = records[rid]
        q, a = _extract_q_and_a(rec)
        if q or a:
            txt = ("Q: " + (q or "") + ("\nA: " + a if a else "")).strip()
        else:
            txt = (_get_text(rec) or "").strip()
        if len(txt) > max_chars:
            txt = txt[:max_chars] + "…"
        blocks.append(f"[B{i}]\n{txt}")
    return "\n\n".join(blocks)


def _blend_semantic_bm25_hits(records: List[Dict[str, Any]],
                              sem_hits: List[Dict[str, Any]],
                              bm25_hits: List[Dict[str, Any]],
                              *,
                              question: Optional[str] = None,
                              max_per_source: int = 2) -> List[Dict[str, Any]]:
    if not sem_hits and not bm25_hits:
        return []

    focus = _question_focus_features(question)
    is_def_question = _question_is_definition(question)
    combined: Dict[int, Dict[str, Any]] = {}

    for h in sem_hits or []:
        rid = h.get("rid")
        if rid is None:
            continue
        combined.setdefault(rid, {"rid": rid, "semantic": 0.0, "bm25": 0.0})
        try:
            combined[rid]["semantic"] = max(combined[rid]["semantic"], float(h.get("score", 0.0)))
        except (TypeError, ValueError):
            continue

    for h in bm25_hits or []:
        rid = h.get("rid")
        if rid is None:
            continue
        combined.setdefault(rid, {"rid": rid, "semantic": 0.0, "bm25": 0.0})
        bm_raw = h.get("score", h.get("score_raw"))
        combined[rid]["bm25"] = max(combined[rid]["bm25"], _normalize_bm25_score(bm_raw))

    ranked: List[Dict[str, Any]] = []
    for rid, feats in combined.items():
        if rid is None or rid < 0 or rid >= len(records):
            continue
        rec = records[rid]
        rec_text = _get_text(rec) or ""
        sem_score = float(feats.get("semantic", 0.0) or 0.0)
        bm_score = float(feats.get("bm25", 0.0) or 0.0)
        base = 0.65 * sem_score + 0.35 * bm_score
        direct_hit = _record_has_direct_answer(rec)
        numeric_hit = focus["numeric_focus"] and _record_matches_numbers(rec, focus["numbers"])
        definition_hit = is_def_question and bool(_DEF_PAT.search(rec_text))
        if direct_hit:
            base += 0.08
        if numeric_hit:
            base += 0.07
        if definition_hit:
            base += 0.1
        if base < 0.05:
            continue
        ranked.append({
            "rid": rid,
            "score": round(base, 6),
            "semantic": sem_score,
            "bm25": bm_score,
            "reasons": {
                "direct_answer": direct_hit,
                "numeric_match": numeric_hit,
                "definition_match": definition_hit,
            }
        })

    ranked.sort(key=lambda x: x["score"], reverse=True)

    if not ranked:
        return []

    seen: Dict[str, int] = {}
    blended: List[Dict[str, Any]] = []
    for item in ranked:
        rid = item["rid"]
        rec = records[rid]
        src_key = f"{rec.get('channel','?')}::{rec.get('file','?')}"
        allowed_per_source = max_per_source + (1 if is_def_question and item.get("reasons", {}).get("definition_match") else 0)
        if seen.get(src_key, 0) >= allowed_per_source:
            continue
        seen[src_key] = seen.get(src_key, 0) + 1
        blended.append(item)
    return blended


def _evidence_score(records: List[Dict[str, Any]], hits: List[Dict[str, Any]]) -> float:
    if not hits:
        return 0.0
    return sum(1 for h in hits if _record_has_direct_answer(records[h["rid"]])) / max(1, len(hits))

def _extract_direct_answer_from_text(text: str) -> Optional[str]:
    """Try to extract a short, direct answer (e.g., Có/Không/Không được…) from a given text block.
    Heuristics favor concise Vietnamese answers commonly appearing in theory Q&A corpora.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    s = text.strip()
    # Pre-clean: drop BM25 labels and question lines to avoid extracting 'Không' from a 'Q:' line
    pre_lines = []
    for ln in s.splitlines():
        if not ln or not ln.strip():
            continue
        t = ln.strip()
        # Drop [B#] labels entirely
        if re.match(r"^\s*\[B\d+\]", t):
            continue
        # Drop lines starting with Q:/Question: (keep answers only)
        if re.match(r"^\s*(Q|Question)\s*:\s*", t, flags=re.IGNORECASE):
            continue
        pre_lines.append(re.sub(r"^\s*(A|Answer)\s*:\s*", "", t, flags=re.IGNORECASE))
    s = "\n".join(pre_lines).strip()
    # Normalize whitespace and split into lines/sentences
    s = re.sub(r"\s+", " ", s)
    # Prefer early short patterns but avoid picking 'Có' when it's part of a noun phrase like 'Có dạng ...'
    patterns = [
        r"\b(tuyệt đối cấm)\b",
        r"^\s*(không được[^\.!?]*)",
        r"^\s*(không[,\.!?]|không\s+[^\.!?]*?)[\.!?]",
        # 'Có' must be followed immediately by punctuation or end, not a word like 'dạng'
        r"^\s*có\s*([,\.!?]|$)",
        r"^\s*(được phép|cho phép)[\.!?]",
    ]
    for pat in patterns:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            ans = m.group(1) if m.lastindex else m.group(0)
            # Map some phrases to canonical short forms
            low = ans.lower()
            if "tuyệt đối cấm" in low or low.startswith("không"):
                return "Không"
            if re.match(r"^có\s*([,\.!?]|$)", low) or "được phép" in low or "cho phép" in low:
                return "Có"
            return ans.strip()
    # Fallback: take the first sentence that begins with 'Không' or 'Có' (after Q-lines removed)
    for seg in re.split(r"(?<=[\.!?])\s+|\n+", s):
        if re.match(r"^\s*không(\s|[,\.\!\?])", seg, flags=re.IGNORECASE):
            return "Không"
        # 'Có' only if followed by punctuation or end
        if re.match(r"^\s*có\s*([,\.\!\?]|$)", seg, flags=re.IGNORECASE):
            return "Có"
    return None

def _extract_best_qna_answer(context_text: str, user_question: Optional[str] = None) -> Optional[str]:
    """Parse Q/A-like blocks inside the context (lines with '?' followed by a non-question line)
    and return the most relevant answer line. Prioritize questions containing keywords like
    'là gì', 'hình dạng', 'mô tả', 'số hiệu', 'ký hiệu'. This avoids misclassifying 'Có dạng ...'
    as a yes/no answer and returns the actual descriptive line.
    """
    if not isinstance(context_text, str) or not context_text.strip():
        return None
    lines = [ln.strip() for ln in context_text.splitlines() if ln.strip()]
    candidates = []

    # Tokenizer for Vietnamese words and simple overlap scoring
    def _tok(s: str) -> List[str]:
        if not s:
            return []
        toks = re.findall(r"[a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹ0-9]+", s.lower())
        stop = {"là", "gì", "thế", "nào", "có", "hay", "không", "được", "bao", "nhiêu", "như", "trong", "khi"}
        return [t for t in toks if len(t) > 1 and t not in stop]

    uq_tokens = set(_tok(user_question or ""))
    for i, ln in enumerate(lines[:-1]):
        if ln.endswith('?'):
            ans = lines[i+1]
            if ans and not ans.endswith('?') and not re.match(r"^\[\w+\]$", ans):
                q = ln.lower()
                score = 0.0
                # Boost definitional-style questions
                for kw in ["là gì", "hình dạng", "mô tả", "số hiệu", "ký hiệu", "khác gì", "đặc điểm", "bao gồm"]:
                    if kw in q:
                        score += 2.0
                # Overlap with user question
                if uq_tokens:
                    q_tokens = set(_tok(q))
                    inter = len(q_tokens & uq_tokens)
                    score += 1.2 * inter
                # Specific alignment for license categories: hạng B vs 'bằng lái b'
                if re.search(r"\bhạng\s*b\b", q) and re.search(r"\b(bằng|giấy\s*phép)\s*lái\s*xe\s*b\b|\bb\b", (user_question or "").lower()):
                    score += 2.5
                # Penalize upgrade/nâng lên when the user asks about vehicles they can drive
                if re.search(r"\bnâng\s+lên\b", q) and re.search(r"\b(lái|điều\s*khiển)\b.*\b(xe)\b", (user_question or "").lower()):
                    score -= 2.0
                # Answer-side heuristics: prefer concrete vehicle/limit descriptions for 'lái xe' intents
                if re.search(r"\b(lái|điều\s*khiển)\b.*\b(xe)\b", (user_question or "").lower()):
                    if re.search(r"\b(chỗ|kg|tấn|tải|rơ\s*moóc|rơmoóc)\b", ans.lower()) or re.search(r"\b\d", ans):
                        score += 1.8
                    # Penalize answers that only list upgrade classes
                    if re.search(r"\b(c1|c|d1|d2)\b", ans.lower()) and not re.search(r"\b(chỗ|kg|tải)\b", ans.lower()):
                        score -= 1.5
                candidates.append((score, ans))
    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[0])
    return candidates[0][1]


def _extract_counts_answer_from_text(text: str, user_question: Optional[str] = None) -> Optional[str]:
    """If the question asks 'bao nhiêu/mấy', look for count lines like
    'Có tổng cộng 28 hành vi ...' or 'Có 14 hành vi ...' and compose a short lead.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    q = (user_question or "").lower()
    if not re.search(r"\b(bao nhiêu|mấy)\b", q):
        return None
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    counts = []
    for ln in lines:
        # Clean Q:/A: prefixes for robust matching/output
        t = re.sub(r"^\s*(Q|Question)\s*:\s*", "", ln, flags=re.IGNORECASE)
        t = re.sub(r"^\s*(A|Answer)\s*:\s*", "", t, flags=re.IGNORECASE)
        low = t.lower()
        # Only consider answer-like lines (after our A:/Q: cleanup this is content lines)
        m = re.search(r"\b(?:có|tổng\s*cộng|bao\s*gồm)\s*(?:tổng\s*cộng\s*)?(\d{1,3})\s+(hành\s*vi|điều|điểm)\b", low, flags=re.IGNORECASE)
        if m:
            num = m.group(1)
            noun = m.group(2)
            # Reconstruct with original casing where possible
            # Keep the whole line but normalize punctuation
            clean = re.sub(r"[;:,，、]+$", "", t).strip()
            counts.append((int(num), noun, clean))
    if not counts:
        return None
    # Prefer the largest total as overall and optionally mention a specific subset
    counts.sort(key=lambda x: -x[0])
    lead_parts = []
    used = set()
    # Add the largest line verbatim
    lead_parts.append(counts[0][2])
    used.add(counts[0][2].lower())
    # If there's a smaller distinct count, append a short clause for it
    for _, _, line in counts[1:]:
        if line.lower() in used:
            continue
        # Use concise append if line contains a qualifier (e.g., 'liên quan trực tiếp ...')
        if len(line) <= 200:
            lead_parts.append(line)
        else:
            # Truncate conservatively
            lead_parts.append(line[:200].rstrip())
        break
    return "; ".join(lead_parts)


def _extract_definition_from_context(text: str) -> Optional[str]:
    """Look for a definitional sentence in the context (contains 'là', 'định nghĩa', 'là gì').
    Return the first candidate sentence that's reasonably short and informative.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    sents = _split_sentences(text)
    for sent in sents:
        low = sent.lower()
        # prefer sentences that explicitly define something
        if re.search(r"\b(là gì|định nghĩa|gọi là|có nghĩa là|là)\b", low):
            # skip question-like lines
            if sent.strip().endswith('?'):
                continue
            # avoid lines that are just labels like '[1]'
            clean = sent.strip()
            if len(clean) > 6 and len(clean) < 400:
                return clean
    return None


def _extract_term_from_question(q: str) -> Optional[str]:
    """Extract the main term from a definitional question like 'phanh tay là gì'.
    Returns a lowercased term without trailing punctuation, or None.
    """
    if not isinstance(q, str) or not q.strip():
        return None
    s = q.strip().lower()
    # Common Vietnamese definitional forms
    m = re.search(r"(.+?)\s+(?:là\s*gì|định\s*nghĩa|có\s*nghĩa\s*là)\b", s)
    if not m:
        return None
    term = m.group(1).strip().strip('"\'\-:;,.!?()[]{}')
    return term or None


def _extract_colon_definition_for_term(context_text: str, term: str) -> Optional[str]:
    """Look for lines/sentences like 'term: ...' and return the sentence as a definition.
    This covers cases where the dataset uses labels and a colon rather than 'là'.
    """
    if not isinstance(context_text, str) or not context_text.strip() or not term:
        return None
    t = term.strip().lower()
    # Split to sentences, but also scan line-wise to catch label prefixes at start of lines
    lines = [ln.strip() for ln in context_text.splitlines() if ln.strip()]
    cand = None
    for ln in lines:
        low = ln.lower()
        # Prefer exact label prefix: 'term:' at start
        if re.match(rf"^\s*{re.escape(t)}\s*[:\-–]", low):
            cand = ln.strip()
            break
    if not cand:
        # Fallback: any sentence containing 'term:'
        for s in _split_sentences(context_text):
            low = s.lower()
            if re.search(rf"\b{re.escape(t)}\s*[:\-–]", low):
                cand = s.strip(); break
    if cand and 6 < len(cand) < 400:
        return cand
    return None


def _build_rich_answer_from_definition(def_sent: str, context_text: str, max_points: int = 4) -> str:
    """Construct a richer answer using the definitional sentence as a lead and
    then add salient bullets and nearby explanatory sentences from the context.
    This avoids returning a single short sentence when more context is available.
    """
    if not def_sent:
        return ""
    parts = [def_sent.strip()]
    # Add salient bullets
    bullets = _extract_key_points(context_text, max_points=max_points)
    if bullets:
        parts.append("")
        for b in bullets:
            # Avoid duplicating the definitional sentence and sanitize bullet text
            b_clean = re.sub(r"\s+", " ", (b or "").strip())
            b_clean = re.sub(r"[;:,，、]+$", "", b_clean).strip()
            if b_clean and b_clean.lower() not in def_sent.strip().lower():
                parts.append(f"- {b_clean}")
    # Extract 1-2 nearby sentences from context that are not duplicates
    sents = [s for s in _split_sentences(context_text) if s.strip() and s.strip().lower() not in def_sent.strip().lower()]
    for extra in sents[:2]:
        if extra and extra.strip() not in parts:
            parts.append("")
            parts.append(extra.strip())
    return "\n".join(parts).strip()

def _is_short_answer(ans: str) -> bool:
    if not isinstance(ans, str):
        return True
    s = ans.strip().lower().strip('.!?:')
    if not s:
        return True
    if s in {"có", "không", "đúng", "sai"}:
        return True
    return len(s) <= 8

def _split_sentences(text: str) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    # Split by sentence end or newlines
    parts = re.split(r"(?<=[\.!?])\s+|\n+", text)
    out = []
    for p in parts:
        t = (p or "").strip()
        if t:
            out.append(t)
    return out

def _strip_bm25_hints(text: str) -> str:
    """Remove BM25 hint section and lines like [B1] ... from a context block.
    Keeps only the main semantic context for answer synthesis.
    """
    if not isinstance(text, str) or not text:
        return text or ""
    s = text
    try:
        s = re.split(r"\[THAM KHẢO BM25\]", s, flags=re.IGNORECASE)[0]
    except Exception:
        pass
    # Drop lines starting with [B<digits>]
    lines = []
    for ln in s.splitlines():
        # Drop BM25 item labels like [B1]
        if re.match(r"^\s*\[B\d+\]", ln):
            continue
        t = (ln or "").rstrip()
        if not t:
            continue
        # Drop question lines entirely (avoid echoing questions in bullets)
        if re.match(r"^\s*(Q|Question)\s*:\s*", t, flags=re.IGNORECASE):
            continue
        # Strip 'A:' prefix but keep the content
        t = re.sub(r"^\s*(A|Answer)\s*:\s*", "", t, flags=re.IGNORECASE)
        lines.append(t)
    return "\n".join(lines)

# ===== Simple conversational memory helpers (per-session last Q/A) =====
_WHY_PAT = re.compile(r"\b(tại sao|vì sao|sao|tại răng|vậy sao)\b", re.IGNORECASE)
_FOLLOWUP_CATCHALL_PAT = re.compile(r"\b(thế còn|vậy còn|cái đó thì sao|thế thì sao|vậy thì sao|như vậy thì sao)\b", re.IGNORECASE)
_EXCEPTION_PAT = re.compile(r"\b(ngoại lệ|trường hợp ngoại lệ)\b", re.IGNORECASE)
_CONFIRM_PAT = re.compile(r"\b(đúng không|phải không|có đúng|okay không|ok không)\b", re.IGNORECASE)

def _get_sess_id(req_session: Optional[str], default_session: str) -> str:
    sid = (req_session or default_session or "api").strip()
    return sid or "api"

def _get_conv(sid: str) -> Dict[str, Any]:
    conv = STATE.get("conv")
    if conv is None:
        conv = {}
        STATE["conv"] = conv
    return conv.setdefault(sid, {"last_q": None, "last_a": None, "turns": []})

def _strip_preface_line(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return ""
    if _PREFACE_LEAD_PAT.match(lines[0]) or lines[0].lower().startswith("vâng, bạn muốn hỏi"):
        lines = lines[1:] if len(lines) > 1 else []
    return ("\n".join(lines)).strip()

_BRACKET_LABEL_PAT = re.compile(r"^\s*\[[A-Za-z0-9]+\]\s*$")
_CHOICE_LINE_PAT = re.compile(r"^\s*[A-DĐÂĂÊÔƠƯa-d]\)\s+.+")
_FILLER_PREFIX_PAT = re.compile(r"^(dưới đây là|sau đây là|cụ thể như sau|chi tiết như sau|tóm lại|trả lời:)\b", re.IGNORECASE)
_HP_AI_PREFIX_PAT = re.compile(r"^hp[_\- ]?ai\s*[:\-]?\s*", re.IGNORECASE)

def _sanitize_llm_output(answer_text: str) -> str:
    """Remove exam-style artifacts like [B1], Q:/A: lines, and multiple-choice options.
    Keep only meaningful sentences/bullets.
    """
    if not isinstance(answer_text, str) or not answer_text.strip():
        return ""
    lines = [l.rstrip() for l in answer_text.splitlines()]
    cleaned: List[str] = []
    for ln in lines:
        s = (ln or "").strip()
        if not s:
            continue
        # Drop bracket-only labels like [B1]
        if _BRACKET_LABEL_PAT.match(s):
            continue
        # Remove leading label like "[B2] text" -> "text"
        s = re.sub(r"^\s*\[[A-Za-z0-9]+\]\s*", "", s)
        # Remove Q:/A: prefixes
        s = re.sub(r"^\s*(Q|A)\s*:\s*", "", s, flags=re.IGNORECASE)
        # Drop HP_AI: prefixes or similar bot tags
        s = _HP_AI_PREFIX_PAT.sub("", s)
        # Drop choice lines like "A) ..."
        if _CHOICE_LINE_PAT.match(s):
            continue
        # Remove filler guide headers like "Dưới đây là ...:" when they are standalone
        if _FILLER_PREFIX_PAT.match(s) and (s.endswith(":") or len(s.split()) <= 6):
            continue
        cleaned.append(s)
    # Collapse possible inline Q/A pairs and remove adjacent duplicates
    collapsed = _collapse_qa_pairs(cleaned)
    # Final cleanup of repetitive spaces
    normalized: List[str] = []
    for c in collapsed:
        if not c or not c.strip():
            continue
        item = re.sub(r"\s+", " ", c).strip()
        # Normalise bullet markers (numbers or • -> "- ")
        item = re.sub(r"^(?:[-•]+)\s*", "- ", item)
        item = re.sub(r"^\(?\d+[\).]\s*", "- ", item)
        if item.startswith("- "):
            # Ensure there is a space after dash and proper trimming
            item = "- " + item[2:].strip()
        normalized.append(item)
    out = "\n".join(normalized)
    return out


def _collapse_qa_pairs(lines: List[str]) -> List[str]:
    """Collapse inline Q/A pairs: if a question line is immediately followed by an answer line,
    drop the question and keep the answer. Also convert short rhetorical Q/A into statements.
    """
    out = []
    i = 0
    n = len(lines)
    while i < n:
        cur = (lines[i] or "").strip()
        nxt = (lines[i+1] or "").strip() if i+1 < n else ""
        # if current looks like a question and next looks like an answer, keep next only
        if cur.endswith("?") and nxt:
            # next is an answer if starts with common Vietnamese answers or dash or 'không'/'có'/'vì'
            if re.match(r"^(không|có|vì|do|bởi|-)\b", nxt, flags=re.IGNORECASE) or nxt.startswith("-"):
                out.append(nxt)
                i += 2
                continue
            # If next is a short 'Không' or 'Có', keep next
            if re.match(r"^(không|có)\b", nxt, flags=re.IGNORECASE):
                out.append(nxt)
                i += 2
                continue
        # Handle patterns like 'Q: ...' / 'A: ...' that survived earlier cleaning
        if re.match(r"^(q\.|q:|question[:\.]?)\s*", cur, flags=re.IGNORECASE) and nxt:
            out.append(nxt)
            i += 2
            continue
        # If current is a short question-like fragment (e.g. 'tại sao không', 'thế còn?') and next
        # looks like an explanatory bullet or begins with dash, drop the question and keep the explanation.
        if len(cur) <= 40 and re.search(r"\b(tại sao|vì sao|thế còn|vậy thì|sao|tại răng)\b", cur, flags=re.IGNORECASE) and nxt:
            if nxt.startswith("-") or len(nxt) > 6:
                out.append(nxt)
                i += 2
                continue
        # Otherwise keep current
        out.append(cur)
        i += 1
    # Remove duplicates of adjacent identical lines
    final = []
    prev = None
    for l in out:
        if l and l == prev:
            continue
        final.append(l)
        prev = l
    return final


def _dedupe_sentences(text: str) -> str:
    """Remove duplicate sentences/lines while preserving order."""
    if not isinstance(text, str) or not text.strip():
        return text or ""
    parts = re.split(r"(?<=[\.\!\?])\s+|\n+", text)
    seen = set()
    out = []
    for p in parts:
        s = (p or "").strip()
        if not s:
            continue
        key = re.sub(r"\s+", " ", s.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return "\n".join(out)


def _strip_internal_headers(text: str) -> str:
    """Remove internal repeated headers/artifacts like 'CÂU HỎI:', '### Trả lời:', and trailing duplicated blocks."""
    if not isinstance(text, str) or not text:
        return text or ""
    s = text
    # Remove 'CÂU HỎI:' and everything after it if it looks like a repeated block
    s = re.split(r"CÂU HỎI:\s*", s, flags=re.IGNORECASE)[0]
    # Remove '### Trả lời:' and trailing repeated content
    s = re.split(r"###\s*Trả\s*lời:\s*", s, flags=re.IGNORECASE)[0]
    # Remove lines starting with 'Lưu ý:' and everything after if it's long
    parts = re.split(r"Lưu ý:\s*", s, flags=re.IGNORECASE)
    if parts:
        s = parts[0]
    # Remove markdown-style headers (lines beginning with #) and common clarifying fragments
    # e.g. '### Câu hỏi của bạn không rõ ràng' or '## CÂU HỎI'
    lines = []
    for line in s.splitlines():
        if not line or not line.strip():
            continue
        # Drop markdown heading lines like '# ...' or '### ...'
        if re.match(r"^\s*#", line):
            continue
        # Drop explicit 'Câu hỏi của bạn' clarifiers
        if re.search(r"câu\s*hỏi\s*của\s*bạn", line, flags=re.IGNORECASE):
            continue
        # Drop 'NỘI DUNG' style labels if left
        if re.match(r"^\s*(nội dung|câu hỏi|trả lời)\b", line, flags=re.IGNORECASE):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _remove_incorrect_modal_transformations(answer_text: str, context_text: str) -> str:
    """Detect and remove sentences where model wrongly converts 'nâng lên' -> 'cho phép điều khiển'."""
    if not isinstance(answer_text, str) or not answer_text.strip():
        return answer_text or ""
    lines = [l.strip() for l in answer_text.splitlines() if l.strip()]
    ctx_low = (context_text or "").lower()
    out = []
    for l in lines:
        low = l.lower()
        # If sentence asserts 'cho phép điều khiển' but context only mentions 'nâng lên', drop it
        if re.search(r"cho phép.*điều khiển", low) and "nâng" in ctx_low and "cho phép" not in ctx_low:
            # skip this hallucinated line
            continue
        out.append(l)
    return "\n".join(out)


def _content_overlap_ratio(answer_text: str, context_text: str) -> float:
    """Simple token-overlap ratio: fraction of answer words that appear in context.
    Returns 0..1. Uses word tokens after basic normalization.
    """
    if not isinstance(answer_text, str) or not answer_text.strip():
        return 0.0
    if not isinstance(context_text, str) or not context_text.strip():
        return 0.0
    aw = re.findall(r"\w+", answer_text.lower())
    cw = set(re.findall(r"\w+", context_text.lower()))
    if not aw:
        return 0.0
    common = sum(1 for t in aw if t in cw)
    return float(common) / float(len(aw))

def _is_yes_no_question(q: str) -> bool:
    if not isinstance(q, str):
        return False
    s = (q or "").strip().lower()
    if not s:
        return False
    if s.endswith("không?"):
        return True
    return bool(re.search(r"\b(có|có thể|được)\b.*\b(không|ko|khôn)\b", s))

def _looks_like_question_output(text: str) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    s = text.strip()
    # If ends with a question mark and is short, likely an echo
    if len(s) <= 120 and s.endswith("?"):
        return True
    # If many question marks relative to length
    qmarks = s.count("?")
    if qmarks >= 2:
        return True
    # If multiple lines that end with '?' (like an echoed Q list)
    lines = [l.strip() for l in s.splitlines() if l.strip()]
    ends_q = sum(1 for l in lines if l.endswith("?"))
    return ends_q >= 2

def _guard_no_qa_echo(answer_text: str, context_text: str, question: Optional[str] = None) -> str:
    """If the output still looks like a question, return a context-only summary instead."""
    if not isinstance(answer_text, str):
        return answer_text
    if not _looks_like_question_output(answer_text):
        return answer_text
    # If context carries a clear polarity (allow/prohibit), prefer a concise yes/no with bullets
    p_ctx = _polarity_from_text(context_text)
    if p_ctx in ("prohibit", "allow"):
        base = "Không" if p_ctx == "prohibit" else "Có"
        support_lines = _format_points_for_question(
            _extract_key_points(context_text, max_points=3),
            question,
            limit=3,
        )
        return "\n".join([base] + support_lines) if support_lines else base
    # Otherwise provide bullet-only summary from context to avoid echoing questions
    support_lines = _format_points_for_question(
        _extract_key_points(context_text, max_points=4),
        question,
        limit=4,
    )
    return "\n".join(support_lines)

def _rewrite_prev_question_why(prev_q: str, last_ans: Optional[str]) -> Optional[str]:
    if not prev_q:
        return None
    q = prev_q.strip().rstrip("?.! ")
    # Pattern: "A có được B hay không"
    m = re.search(r"^(.+?)\s+có\s+được\s+(.+?)\s+hay\s+không$", q, flags=re.IGNORECASE)
    if m:
        a = m.group(1).strip()
        b = m.group(2).strip()
        neg = (last_ans or "").strip().lower().startswith("không")
        phrase = "không được" if neg else "được"
        # Build: "Tại sao [không được|được] B khi A?"
        return f"tại sao {phrase} {b} khi {a}?"
    # Generic fallback
    return f"tại sao như vậy về: {q}?"

def _rewrite_from_turn(turn: Dict[str, Any], mode: str) -> Optional[str]:
    """Build a resolved follow-up based on a specific turn and mode: why|exception|confirm|generic"""
    base_q = (turn.get("resolved") or turn.get("q") or "").strip()
    if not base_q:
        return None
    if mode == "why":
        ans = (turn.get("a") or "").strip().lower()
        return _rewrite_prev_question_why(base_q, ans)
    if mode == "exception":
        return f"có ngoại lệ nào đối với: {base_q} không?"
    if mode == "confirm":
        return f"kết luận cho câu trước có chính xác không về: {base_q}?"
    # generic
    return f"bổ sung thông tin cho: {base_q}?"

def _resolve_followup_with_memory(current_q: str, sid: str) -> str:
    """If the question is short/elliptical and looks like a follow-up, expand it using last Q/A."""
    try:
        cq = (current_q or "").strip()
        if not cq:
            return cq
        conv = _get_conv(sid)
        turns = list(conv.get("turns") or [])
        # Prefer to leverage a short window of history for various follow-up intents
        if len(cq) <= 24:
            mode = None
            if _WHY_PAT.search(cq):
                mode = "why"
            elif _EXCEPTION_PAT.search(cq):
                mode = "exception"
            elif _CONFIRM_PAT.search(cq):
                mode = "confirm"
            elif _FOLLOWUP_CATCHALL_PAT.search(cq):
                mode = "generic"
            if mode and turns:
                for turn in reversed(turns):
                    rew = _rewrite_from_turn(turn, mode)
                    if rew:
                        return rew
        # Legacy: WHY based only on last_q/last_a
        if len(cq) <= 24 and _WHY_PAT.search(cq):
            last_q, last_a = conv.get("last_q"), conv.get("last_a")
            if last_q:
                rew = _rewrite_prev_question_why(last_q, last_a)
                if rew:
                    return rew
        return cq
    except Exception:
        return current_q

def _update_conv_memory(sid: str, resolved_question: str, answer_text: str, orig_question: Optional[str] = None):
    try:
        conv = _get_conv(sid)
        rq = (resolved_question or "").strip()
        a = _strip_preface_line(answer_text or "")
        conv["last_q"], conv["last_a"] = rq, a
        turn = {"q": rq, "a": a}
        if orig_question:
            turn["orig"] = (orig_question or "").strip()
        conv.setdefault("turns", []).append(turn)
        depth = int(getattr(STATE.get("args", SimpleNamespace()), "conv_memory_depth", 4) or 4)
        if len(conv["turns"]) > depth:
            conv["turns"] = conv["turns"][-depth:]
    except Exception:
        pass

def _extract_key_points(context_text: str, max_points: int = 4, max_len: int = 420) -> List[str]:
    """Pick salient sentences from context for bullet points.
    Prioritize safety reasons, rules, and operational guidance commonly present in the dataset.
    """
    def _extract_numbered_items(text: str, min_items: int = 2, max_items: int = 20) -> List[str]:
        if not isinstance(text, str) or not text.strip():
            return []
        lines = [ln.strip() for ln in text.splitlines() if ln and ln.strip()]
        blocks: List[List[str]] = []
        current: List[str] = []

        def _flush_block():
            nonlocal current
            if current:
                blocks.append(current)
                current = []

        for line in lines:
            matches = list(re.finditer(r"(\d{1,2})\.\s*", line))
            if not matches:
                _flush_block()
                continue

            line_end = len(line)
            for idx, m in enumerate(matches):
                start = m.end()
                end = matches[idx + 1].start() if idx + 1 < len(matches) else line_end
                chunk = line[start:end].strip(" ,;.-")
                chunk = re.sub(r"\s+", " ", chunk)
                if not chunk:
                    continue
                num = m.group(1)
                if not chunk.startswith(f"{num}."):
                    chunk = f"{num}. {chunk}"
                current.append(chunk)
                if len(current) >= max_items:
                    break
            if len(current) >= max_items:
                break
        _flush_block()

        if not blocks:
            return []
        best = max(blocks, key=len)
        if len(best) < min_items:
            return []

        cleaned_items: List[str] = []
        for item in best[:max_items]:
            item_clean = re.sub(r"\s+", " ", item.strip())
            if not re.search(r"[\.\!\?…]$", item_clean):
                item_clean += "."
            cleaned_items.append(item_clean)
        return cleaned_items

    # Strip BM25 hints to avoid pulling hint lines into bullets
    base_text = _strip_bm25_hints(context_text)
    numbered = _extract_numbered_items(base_text, max_items=max(max_points, 12))
    if numbered:
        return numbered
    sents = _split_sentences(base_text)
    if not sents:
        return []
    # Helper: split procedural sentences into meaningful clauses for clearer bullets
    def _split_clauses(sentence: str) -> List[str]:
        if not isinstance(sentence, str) or not sentence.strip():
            return []
        s = sentence.strip()
        # Prefer splitting on strong step separators; avoid naive comma / 'và' splits that create terse fragments
        parts = re.split(r"\s*(?:;|\bsau đó\b|\btiếp theo\b|\btrước hết\b|\bcuối cùng\b)\s*", s, flags=re.IGNORECASE)
        parts = [p.strip().strip('-•') for p in parts if p and p.strip()]
        if not parts:
            return [s]
        # Merge too-short fragments (e.g., 'Khi lên xe') with the following clause
        merged: List[str] = []
        i = 0
        while i < len(parts):
            cur = parts[i]
            # measure by word count
            wc = len(re.findall(r"\w+", cur))
            if wc < 5 and i + 1 < len(parts):
                nxt = parts[i + 1]
                cur = (cur.rstrip(',') + ", " + nxt.lstrip(', ')).strip()
                i += 2
            else:
                i += 1
            # Cap excessive length to keep bullets readable
            if len(cur) > 360:
                cur = cur[:360].rstrip()
            merged.append(cur)
        return merged or [s]
    # Scoring by keyword presence
    keywords = [
        (2.5, ["tuyệt đối cấm", "cấm vượt", "không được vượt", "không được" ]),
        (2.0, ["không gian hẹp", "tầm nhìn", "va chạm", "nguy hiểm", "tai nạn"]),
        (1.8, ["biển báo", "tuân thủ", "chỉ dẫn"]),
        (1.5, ["phạt", "trừ điểm", "vi phạm"]),
        (1.2, ["khoảng cách", "tốc độ", "chiếu sáng", "thông gió", "camera"]),
    ]
    scored = []
    for sent in sents:
        if len(sent) < 6:
            continue
        # Skip question-like sentences and bracket-only lines
        if sent.endswith('?'):
            continue
        if _BRACKET_LABEL_PAT.match(sent):
            continue
        score = 0.0
        low = sent.lower()
        for w, kws in keywords:
            if any(kw in low for kw in kws):
                score += w
        # Light bonus if mentions the exact location/topic
        if "hầm" in low and "vượt" in low:
            score += 2.0
        if score > 0:
            # If sentence is lengthy, split into clauses to avoid '…' truncation
            clauses = _split_clauses(sent.strip())
            if len(clauses) == 1:
                txt = clauses[0]
                scored.append((score, txt))
            else:
                # Add up to 3 clauses with slight score decay to preserve order
                decay = 0.0
                for i, cl in enumerate(clauses[:3]):
                    scored.append((max(0.0, score - decay), cl))
                    decay += 0.2
    if not scored:
        # Fallback: pick first few non-empty, non-question, non-label sentences (no ellipsis)
        fallback = []
        for s in sents:
            if not s or s.endswith('?'):
                continue
            if _BRACKET_LABEL_PAT.match(s):
                continue
            clauses = _split_clauses(s)
            for cl in clauses:
                fallback.append(cl)
                if len(fallback) >= max_points:
                    break
            if len(fallback) >= max_points:
                break
        return fallback[:max_points]
    scored.sort(key=lambda x: -x[0])
    # Deduplicate by lowercase form
    seen, out = set(), []
    for _, txt in scored:
        key = txt.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(txt)
        if len(out) >= max_points:
            break
    # Final cleaning to avoid empty or oddly punctuated bullets
    cleaned: List[str] = []
    for t in out:
        s = re.sub(r"\s+", " ", (t or "").strip())
        s = re.sub(r"[;:,，、]+$", "", s).strip()
        # Capitalize the first letter if it's lowercase for better readability
        if s and s[0].islower():
            try:
                s = s[0].upper() + s[1:]
            except Exception:
                pass
        # Normalize bullet termination
        if s and not re.search(r"[\.\!\?…]$", s):
            s = s + "."
        if s:
            cleaned.append(s)
    return cleaned


_KNOWN_LICENSE_CODES: Set[str] = {
    "A", "A0", "A1", "A2", "A3", "A4",
    "B", "B11", "B12", "B1", "B2",
    "C", "C1", "C2",
    "D", "D1", "D2",
    "E", "E1", "E2",
    "F", "FA", "FB", "FC", "FD", "FE",
}

_LICENSE_TOKEN_RE = re.compile(r"\b([A-Za-z]{1,2}\d?)\b")


def _clean_point_text(text: Optional[str]) -> str:
    if not isinstance(text, str):
        return ""
    cleaned = re.sub(r"\s+", " ", text.strip())
    cleaned = re.sub(r"[;:,，、]+$", "", cleaned).strip()
    return cleaned


def _extract_license_classes(text: Optional[str]) -> Set[str]:
    if not isinstance(text, str) or not text.strip():
        return set()
    tokens = _LICENSE_TOKEN_RE.findall(text)
    return {tok.upper() for tok in tokens if tok and tok.upper() in _KNOWN_LICENSE_CODES}


def _point_mentions_license_class(text: str, license_codes: Set[str]) -> bool:
    if not license_codes:
        return True
    tokens = {tok.upper() for tok in _LICENSE_TOKEN_RE.findall(text)}
    if tokens & license_codes:
        return True
    for code in license_codes:
        base = code.rstrip("0123456789")
        if base and re.search(rf"\bhạng\s+{re.escape(base)}\b", text, flags=re.IGNORECASE):
            return True
    return False


def _select_supporting_points(points: Iterable[str], question: Optional[str], limit: Optional[int] = None) -> List[str]:
    cleaned: List[str] = []
    seen: Set[str] = set()
    for raw in points:
        text = _clean_point_text(raw)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
    if not cleaned:
        return []
    license_codes = _extract_license_classes(question)
    if license_codes:
        filtered = [pt for pt in cleaned if _point_mentions_license_class(pt, license_codes)]
        base = filtered if filtered else cleaned
    else:
        base = cleaned
    if limit is not None and limit >= 0:
        base = base[:limit]
    return base


def _format_points_for_question(points: Iterable[str], question: Optional[str], limit: Optional[int] = None) -> List[str]:
    selected = _select_supporting_points(points, question, limit)
    return [f"- {pt}" for pt in selected]


_REF_BLOCK_PAT = re.compile(r"\[(\d+)\]\s*(.*?)(?=\n\[\d+\]|\Z[n])", re.DOTALL)


_CITATION_PAT = re.compile(r"\(tham khảo \[\d+\]\)", re.IGNORECASE)


def _marks_reasoned_sections(answer: Optional[str]) -> bool:
    if not answer:
        return False
    normalized = answer.lower()
    if "suy luận hợp lý" in normalized:
        return True
    return bool(_CITATION_PAT.search(answer))


def _find_reference_label(fragment: str, context_text: str) -> Optional[str]:
    if not fragment or not context_text:
        return None
    frag_core = re.sub(r"^[-–•]\s*", "", fragment).strip()
    if not frag_core:
        return None
    if "tham khảo [" in frag_core.lower():
        return None
    frag_norm = re.sub(r"\s+", " ", frag_core.lower())
    frag_words = [w for w in re.findall(r"[\wÀ-ỹ]+", frag_norm) if len(w) > 2]
    for match in _REF_BLOCK_PAT.finditer(context_text or ""):
        label = match.group(1)
        block = re.sub(r"\s+", " ", (match.group(2) or "").lower())
        if frag_norm and frag_norm in block:
            return f"[{label}]"
        if frag_words:
            hits = sum(1 for w in frag_words if w in block)
            if hits >= max(2, len(frag_words) // 2 + 1):
                return f"[{label}]"
    return None


def _attach_reference_label(text: str, context_text: str) -> str:
    if not isinstance(text, str):
        return text
    if "tham khảo [" in text.lower():
        return text
    ref = _find_reference_label(text, context_text)
    if not ref:
        return text
    stripped = text.rstrip()
    if stripped.endswith("."):
        return stripped[:-1].rstrip() + f" (tham khảo {ref})."
    if stripped.endswith((";", ":", ",")):
        return stripped + f" (tham khảo {ref})"
    return stripped + f" (tham khảo {ref})"


def _enrich_short_answer(answer_text: str, context_text: str, max_points: int = 3) -> str:
    """If the answer is too short (e.g., just 'Có'/'Không'), enrich it with 2-3 bullets from context."""
    base = (answer_text or "").strip()
    # Decide whether enrichment is needed: very short OR single-sentence concise lead
    need_enrich = _is_short_answer(base)
    if not need_enrich:
        ans_sents = _split_sentences(base)
        if len(ans_sents) == 1 and len(base) <= 180:
            need_enrich = True
    if not need_enrich:
        return base
    # Try to extract a short definitional sentence from context when answer looks like a definition request
    sents = _split_sentences(context_text)
    def_sent = None
    for sent in sents:
        if re.search(r"\b(là gì|là|định nghĩa|gọi là|có nghĩa là)\b", sent, flags=re.IGNORECASE):
            if sent.strip().endswith('?'):
                continue
            def_sent = sent.strip()
            break
    bullets = _extract_key_points(context_text, max_points=max_points)
    # If we have a definitional sentence and the base answer is not a simple Có/Không, prefer that as lead
    lines: List[str] = []
    if def_sent and base and base.lower() not in {"có", "không", "đúng", "sai"}:
        # Put base (if any) as very short lead, then add definitional sentence and bullets
        lines.append(base)
        lines.append(_attach_reference_label(def_sent, context_text))
    elif def_sent and not base:
        lines.append(_attach_reference_label(def_sent, context_text))
    else:
        if base:
            lines.append(base)
    base_low = base.lower()
    for b in bullets:
        # Avoid duplicating a sentence that is the same as def_sent
        b_norm = (b or "").strip()
        if def_sent and b_norm.lower() == def_sent.lower():
            continue
        # Avoid duplicating the base sentence itself
        if b_norm.lower() == base_low:
            continue
        # Sanitize bullet text and skip empties
        b_clean = re.sub(r"\s+", " ", b_norm)
        b_clean = re.sub(r"[;:,，、]+$", "", b_clean).strip()
        if not b_clean:
            continue
        bullet_with_ref = _attach_reference_label(f"- {b_clean}", context_text)
        lines.append(bullet_with_ref)
    return "\n".join(lines).strip()

# ===== Grounding and contradiction detection utilities =====
_ALLOW_PATTERNS = [
    r"\bđược phép\b", r"\bvẫn được\b", r"\bđược lái\b", r"\bđược điều khiển\b"
]
_PROHIBIT_PATTERNS = [
    r"\bnghiêm cấm\b", r"\bbị nghiêm cấm\b", r"\bkhông được\b", r"\bcấm\b", r"\btuyệt đối cấm\b",
    r"\bcấm điều khiển\b", r"\bcấm lái\b", r"\bvi phạm\b",
    r"(^|\n)\s*không\s+(?:được\s+)?(?:lái|điều khiển|vượt|dừng|đỗ|lùi|quay\s+đầu|đổi\s+làn)\b",
]

def _polarity_from_text(text: str) -> Optional[str]:
    if not isinstance(text, str) or not text.strip():
        return None
    s = text.lower()
    allow = any(re.search(p, s) for p in _ALLOW_PATTERNS)
    prohibit = any(re.search(p, s) for p in _PROHIBIT_PATTERNS)
    if prohibit and not allow:
        return "prohibit"
    if allow and not prohibit:
        return "allow"
    return None


_NUM_UNIT_RE = re.compile(
    r"\b\d+(?:[\.,]\d+)?\s*(?:mg\/?l(?:ít)?(?:\s*khí\s*thở)?|mg\/?l|ml|%|chỗ(?:\s*ngồi)?|kg|kilog?ram|tấn|tan|t|km(?:\/?h|\/?gi(?:ờ|o))?|kmh|km|m(?:\/s)?|gi(?:ờ|o)|phút|giây)\b",
    re.IGNORECASE,
)

# Basic mapping for common Vietnamese spelled-out numbers to digits. This covers 0..20 and common
# decades up to 100. It's intentionally small and conservative to avoid false positives.
_VN_NUMBER_WORDS = {
    "không": "0",
    "một": "1", "mốt": "1",
    "hai": "2",
    "ba": "3",
    "bốn": "4", "tư": "4",
    "năm": "5", "lăm": "5",
    "sáu": "6",
    "bảy": "7", "bẩy": "7",
    "tám": "8",
    "chín": "9",
    "mười": "10",
    "mười một": "11", "mười một": "11",
    "mười hai": "12",
    "mười ba": "13",
    "mười bốn": "14",
    "mười lăm": "15",
    "mười sáu": "16",
    "mười bảy": "17",
    "mười tám": "18",
    "mười chín": "19",
    "hai mươi": "20", "hai mươi": "20",
    "ba mươi": "30",
    "bốn mươi": "40",
    "năm mươi": "50",
    "sáu mươi": "60",
    "bảy mươi": "70",
    "tám mươi": "80",
    "chín mươi": "90",
    "một trăm": "100",
}

def _normalize_vn_number_words(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text or ""
    s = text.lower()
    units_map = {
        'không': 0, 'một': 1, 'mốt': 1, 'hai': 2, 'ba': 3, 'bốn': 4, 'tư': 4,
        'năm': 5, 'lăm': 5, 'sáu': 6, 'bảy': 7, 'tám': 8, 'chín': 9
    }

    def word_to_int(w: str) -> Optional[int]:
        return units_map.get(w.strip(), None)

    # Handle patterns like 'một trăm hai mươi lăm', 'hai mươi mốt', 'mười hai'
    def replace_compound(match: re.Match) -> str:
        grp = match.group(0)
        parts = re.findall(r"[a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹ\w]+", grp)
        total = 0
        i = 0
        # Look for 'trăm'
        if 'trăm' in parts:
            try:
                idx = parts.index('trăm')
                if idx > 0 and word_to_int(parts[idx-1]) is not None:
                    total += word_to_int(parts[idx-1]) * 100
                    # consume up to idx
                    rest = parts[idx+1:]
                else:
                    # assume 1 trăm
                    total += 100
                    rest = parts[idx+1:]
            except Exception:
                rest = parts
        else:
            rest = parts

        # handle tens and units in rest
        if rest:
            # pattern: 'mươi' presence
            if 'mươi' in rest:
                try:
                    idx = rest.index('mươi')
                    tens_candidate = rest[idx-1] if idx-1 >= 0 else None
                    tens = word_to_int(tens_candidate) if tens_candidate else None
                    if tens is not None:
                        total += tens * 10
                    else:
                        # leading 'mươi' (like 'mươi' alone) -> 10
                        total += 10
                    # unit after mươi
                    if idx+1 < len(rest):
                        u = rest[idx+1]
                        ui = word_to_int(u)
                        if ui is not None:
                            # 'mốt' and 'lăm' special handling handled by map
                            total += ui
                except Exception:
                    pass
            elif rest[0] == 'mười' or (rest[0].startswith('mười')):
                # 'mười' + unit
                total += 10
                if len(rest) > 1:
                    ui = word_to_int(rest[1])
                    if ui is not None:
                        total += ui
            else:
                # maybe single unit
                ui = word_to_int(rest[0])
                if ui is not None:
                    total += ui

        if total == 0:
            return grp
        return str(total)

    # Replace compound phrases first (look for sequences of number-words up to length 6)
    pattern = r"\b(?:(?:một|mốt|hai|ba|bốn|tư|năm|lăm|sáu|bảy|tám|chín|mười|mươi|trăm|không|hai mươi|ba mươi|bốn mươi|năm mươi|sáu mươi|bảy mươi|tám mươi|chín mươi)(?:\s+|\-)?){1,6}\b"
    try:
        s = re.sub(pattern, replace_compound, s, flags=re.IGNORECASE)
    except Exception:
        pass

    # Replace remaining individual number words from the conservative map
    for word in sorted(_VN_NUMBER_WORDS.keys(), key=lambda x: -len(x)):
        digit = _VN_NUMBER_WORDS[word]
        s = re.sub(r"\b" + re.escape(word) + r"\b", digit, s, flags=re.IGNORECASE)
    return s


def _find_numbers_with_units(text: str) -> List[str]:
    if not isinstance(text, str) or not text:
        return []
    norm = _normalize_vn_number_words(text)
    return [m.group(0) for m in _NUM_UNIT_RE.finditer(norm)]

def _numbers_not_in_context(answer_text: str, context_text: str) -> List[str]:
    ans_nums = _find_numbers_with_units(answer_text)
    if not ans_nums:
        return []
    ctx_low = (context_text or "").lower()
    missing = []
    for token in ans_nums:
        if token.lower() not in ctx_low:
            missing.append(token)
    return missing


_GAP_NUMERIC_PAT = re.compile(r"\b(bao nhiêu|mấy|bao lâu|bao xa|bao nhiêu phần trăm|tối đa|tối thiểu)\b", re.IGNORECASE)
_GAP_CONDITION_PAT = re.compile(r"\b(khi nào|trong trường hợp nào|lúc nào|bao giờ|thời điểm nào)\b", re.IGNORECASE)


def _should_allow_reasoned_gap(question: str, answer_text: str, context_text: str) -> bool:
    if not question or not answer_text:
        return False
    ans_clean = (answer_text or "").strip()
    if not ans_clean or len(ans_clean.split()) < 4:
        return False
    q_low = (question or "").lower()
    if _GAP_NUMERIC_PAT.search(q_low):
        missing_nums = _numbers_not_in_context(answer_text, context_text)
        if missing_nums:
            return True
    if _GAP_CONDITION_PAT.search(q_low):
        ctx_low = (context_text or "").lower()
        # Allow reasoning when context lacks clear trigger phrases (chưa có 'khi', 'trường hợp' lặp lại)
        trigger_hits = ctx_low.count("khi") + ctx_low.count("trường hợp")
        if trigger_hits < 2:
            return True
    return False


def _build_reasoned_gap_answer(answer_text: str, context_text: str) -> str:
    base = (answer_text or "").strip()
    if not base:
        return ""

    lead_lines: List[str] = []
    ctx_summary = "Chưa thấy nội dung tham khảo nêu trực tiếp câu trả lời." if context_text else "Không có nội dung tham khảo phù hợp để trả lời trực tiếp."
    lead_lines.append(ctx_summary)

    points = _extract_key_points(context_text, max_points=2) if context_text else []
    for pt in points:
        clean = re.sub(r"^\s*[-–•]\s*", "", (pt or "").strip())
        if not clean:
            continue
        labeled = _attach_reference_label(clean, context_text)
        if labeled.lower().startswith("suy luận hợp lý"):
            lead_lines.append(labeled)
            continue
        if labeled.startswith("-"):
            lead_lines.append(labeled)
        else:
            lead_lines.append(f"- {labeled}")

    reason_block = base
    if "\n" in reason_block:
        if not reason_block.lower().startswith("suy luận hợp lý"):
            reason_block = "Suy luận hợp lý:\n" + reason_block
    else:
        if not reason_block.lower().startswith("suy luận hợp lý"):
            reason_block = f"Suy luận hợp lý: {reason_block}"

    lead_lines.append(reason_block.strip())

    note_line = "Ghi chú: Đây là suy luận dựa trên kiến thức lái xe phổ biến; hãy đối chiếu với quy định hiện hành trước khi áp dụng."
    if note_line.lower() not in reason_block.lower():
        lead_lines.append(note_line)

    return "\n".join([ln for ln in lead_lines if ln]).strip()

def _ground_if_contradiction(answer_text: str, context_text: str, question: str) -> Dict[str, Any]:
    """Detect contradictions between context and answer; if found, rewrite answer using context only.
    Returns dict with keys: text, grounded(bool), reason(str), missing_numbers(list)
    """
    ans = (answer_text or "").strip()
    ctx = (context_text or "").strip()
    if not ans or not ctx:
        return {"text": ans, "grounded": False, "reason": None, "missing_numbers": []}
    p_ctx = _polarity_from_text(ctx)
    p_ans = _polarity_from_text(ans)
    missing_nums = _numbers_not_in_context(ans, ctx)
    reasons = []
    if p_ctx == "prohibit" and p_ans == "allow":
        reasons.append("contradiction: context=prohibit, answer=allow")
    if p_ctx == "allow" and p_ans == "prohibit":
        reasons.append("contradiction: context=allow, answer=prohibit")
    if missing_nums:
        reasons.append("unsupported_numbers")

    if reasons and p_ctx in ("prohibit", "allow"):
        # Rewrite succinctly according to context polarity and include 2-3 bullets from context
        base = "Không" if p_ctx == "prohibit" else "Có"
        bullets = _extract_key_points(ctx, max_points=3)
        lines = [base]
        for b in bullets:
            lines.append(f"- {b}")
        return {
            "text": "\n".join(lines).strip(),
            "grounded": True,
            "reason": ", ".join(reasons),
            "missing_numbers": missing_nums,
        }

    return {"text": ans, "grounded": False, "reason": None, "missing_numbers": missing_nums}
    
def _sanitize_model_output(answer: str, question: str, context: str) -> str:
    """Remove <CONTEXT>/<QUESTION> blocks, exact question echoes, and header lines if model echoed them."""
    if not isinstance(answer, str):
        return ""
    s = answer
    try:
        s = re.sub(r"(?is)<CONTEXT>.*?</CONTEXT>", "", s)
        s = re.sub(r"(?is)<QUESTION>.*?</QUESTION>", "", s)
        # Remove header lines like 'NỘI DUNG LÝ THUYẾT THAM KHẢO' or 'CÂU HỎI'
        s = re.sub(r"(?im)^NỘI DUNG LÝ THUYẾT THAM KHẢO\s*:?.*$", "", s)
        s = re.sub(r"(?im)^CÂU HỎI\s*:?.*$", "", s)
        # Remove exact question line if echoed
        q = (question or "").strip()
        if len(q) >= 4:
            q_esc = re.escape(q)
            s = re.sub(r"(?im)^\s*" + q_esc + r"\s*$", "", s)
            s = re.sub(r"(?im)^\s*Q\s*[:：]\s*" + q_esc + r"\s*$", "", s)
        s = s.strip()
        s = re.sub(r"\n{3,}", "\n\n", s)
    except Exception:
        s = answer.strip()
    return s


def _strip_artifact_wrappers(answer_text: str) -> str:
    """Remove redundant square/curly bracket wrappers and merge split numbered lines."""
    if not isinstance(answer_text, str):
        return ""
    s = answer_text.replace("{", "").replace("}", "")
    # Normalize enumerations like "[1.\n..." or "[1. ..."
    s = re.sub(r"\[(\d+)\.\s*\n", r"\1. ", s)
    s = re.sub(r"\[(\d+)\.\s*", r"\1. ", s)
    # Remove empty brackets and leftover closers
    s = re.sub(r"\[\s*\]", "", s)
    s = s.replace("[", "").replace("]", "")

    raw_lines = s.splitlines()
    merged: List[str] = []
    i = 0
    while i < len(raw_lines):
        cur = (raw_lines[i] or "").strip()
        if not cur:
            merged.append("")
            i += 1
            continue
        if re.match(r"^\d+\.$", cur) and i + 1 < len(raw_lines):
            nxt = (raw_lines[i + 1] or "").strip()
            combined = f"{cur} {nxt}".strip()
            merged.append(combined)
            i += 2
            continue
        merged.append(cur)
        i += 1

    cleaned_lines: List[str] = []
    prev_blank = False
    for line in merged:
        line = line.strip()
        if not line:
            if not prev_blank:
                cleaned_lines.append("")
            prev_blank = True
            continue
        normalized = re.sub(r"\s+", " ", line)
        cleaned_lines.append(normalized.strip())
        prev_blank = False

    return "\n".join(cleaned_lines).strip()

def build_simple_vietnamese_prompt(question: str, context: str, sample_answer: str = "", verbosity: str = "normal") -> str:
    """Build the same simple Vietnamese prompting system from main.py with richer guidance."""
    verbosity = (verbosity or "normal").lower()
    detail_rules = {
        "minimal": (
            "- Trả lời ngắn gọn bằng 1 câu kết luận rõ ràng (ghi rõ Có/Không hoặc nội dung trọng tâm).\n"
            "- Sau đó thêm 1-2 gạch đầu dòng tóm lược điều kiện, số liệu hoặc lưu ý an toàn thiết yếu.\n"
            "- Nếu cần suy luận, thêm dòng 'Suy luận hợp lý:' kèm lý do ngắn gọn và nhắc kiểm chứng văn bản chính thức.\n"
            "- Tránh nhắc lại nguyên văn câu hỏi hoặc liệt kê dài dòng."
        ),
        "detailed": (
            "- Trình bày thành 3 phần rõ ràng:\n"
            "  1) **Tóm tắt**: 1-2 câu khẳng định nội dung chính, nêu kết luận sớm.\n"
            "  2) **Chi tiết chính**: ≥3 gạch đầu dòng, mỗi bullet mô tả điều kiện, ví dụ, hoặc số liệu; thêm '(tham khảo [n])' khi biết đoạn nguồn tương ứng.\n"
            "  3) **Suy luận hợp lý** (nếu cần): giải thích mở rộng, ví dụ thực tế hoặc lưu ý an toàn, đồng thời nhắc kiểm chứng quy định hiện hành.\n"
            "- Gộp các đoạn trùng lặp, ưu tiên ý quan trọng trước và giữ giọng văn mạch lạc, dễ đọc."
        ),
    }
    detail_guidance = detail_rules.get(
        verbosity,
        (
            "- Mở đầu bằng 1 câu trả lời trực tiếp nêu rõ kết luận chính.\n"
            "- Có thể suy luận mở rộng nhưng phải chỉ rõ phần nào là 'Suy luận hợp lý:' và nhắc người đọc kiểm chứng quy định hiện hành.\n"
            "- Sau đó liệt kê 2-3 gạch đầu dòng diễn giải điều kiện, số liệu hoặc lưu ý quan trọng; thêm '(tham khảo [n])' khi xác định được đoạn nguồn.\n"
            "- Diễn đạt lại ý bằng ngôn từ thân thiện, tránh sao chép nguyên văn dài dòng từ tham khảo."
        ),
    )

    core_rules = (
        "- Trước khi trả lời, rà soát toàn bộ tham khảo để tìm tối đa 3 ý chính, số liệu then chốt và mối liên hệ với câu hỏi (bước phân tích nội bộ, không in ra).\n"
        "- Khi trả lời, giữ lập luận bám sát tài liệu; mọi suy luận thêm phải gắn nhãn 'Suy luận hợp lý:' và không mâu thuẫn với tham khảo.\n"
        "- Không khẳng định thông tin trái với tài liệu tham khảo; nếu thiếu căn cứ phải nói rõ trước khi suy luận.\n"
        "- Khi trích dẫn lại số liệu/điều kiện, cố gắng chỉ ra `(tham khảo [n])` theo số thứ tự block ngữ cảnh tương ứng. Không tự tạo số tham khảo nếu không chắc chắn.\n"
        "- Nếu thiếu dữ kiện, nêu rõ phần thiếu trước, sau đó mới đưa khuyến nghị tiếp theo.\n"
    )

    system_prompt = (
        "Bạn là trợ lý AI chuyên về LÝ THUYẾT LÁI XE Ô TÔ tại Việt Nam.\n\n"
        "Nhiệm vụ:\n"
        "- Nhận vào 2 tham số:\n"
        "  1) CÂU HỎI của người dùng.\n"
        "  2) NỘI DUNG LÝ THUYẾT THAM KHẢO (đã được trích xuất từ tài liệu chuẩn).\n\n"
        "Quy tắc quan trọng:\n"
        f"{core_rules}"
        "- Văn phong: tiếng Việt, thân thiện, dễ hiểu, không châm biếm.\n"
        "- Không nhắc lại nguyên văn câu hỏi và không đề cập tới 'prompt' hay hướng dẫn hệ thống.\n"
        "- Không thêm mục 'Nguồn:' hay 'Tài liệu tham khảo' ở cuối câu trả lời; chỉ gắn '(tham khảo [n])' trong câu khi cần.\n"
        "- Chỉ trả về phần trả lời thuần văn bản (không JSON, không thêm nhãn hệ thống).\n"
        f"- Định dạng khi trả lời:\n{detail_guidance}"
    )

    if sample_answer and sample_answer.strip():
        system_prompt += (
            "\nVí dụ về phong cách trả lời mong muốn (chỉ tham khảo, không được sao chép chữ y chang):\n"
            f"---\n{sample_answer.strip()}\n---"
        )

    clean_question = (question or "").strip()
    clean_context = (context or "").strip()
    prompt_parts = [system_prompt]
    if clean_context:
        prompt_parts.append(f"\nNỘI DUNG LÝ THUYẾT THAM KHẢO:\n{clean_context}")
    else:
        prompt_parts.append(f"\nNỘI DUNG LÝ THUYẾT THAM KHẢO:\n(Không có nội dung tham khảo)")
    prompt_parts.append(f"\nCÂU HỎI:\n{clean_question}")
    return "\n".join(prompt_parts)

def simple_generate_answer(llm, question: str, context: str, max_new_tokens: int = 512, temperature: float = 0.3, max_prompt_chars: int = 12000, verbosity: str = 'normal') -> str:
    """Simpler generate that mirrors main.py: build prompt, truncate context if needed, call llm, sanitize echo.
    Accepts verbosity: 'minimal'|'normal'|'detailed' to influence prompt building."""
    try:
        if not hasattr(llm, 'generate'):
            return "LLM không khả dụng"

        # Build prompt using centralized builder with verbosity
        simple_prompt = build_simple_vietnamese_prompt(question, context, verbosity=verbosity)

        # Truncate prompt if too long (keep more context if possible)
        if len(simple_prompt) > max_prompt_chars:
            keep_ctx = max(1000, min(len(context or ""), max_prompt_chars - 1500))
            context_trunc = (context or "")[:keep_ctx] + ("..." if len(context or "") > keep_ctx else "")
            simple_prompt = build_simple_vietnamese_prompt(question, context_trunc, verbosity=verbosity)

        if hasattr(llm, 'set_params'):
            llm.set_params(temperature=temperature, max_new_tokens=max_new_tokens)

        stop_tokens = ["<|endoftext|>", "</s>"]
        answer = llm.generate(simple_prompt, stop=stop_tokens)

        if answer:
            # Remove common leading phrases and sanitize echoes
            clean_patterns = [
                r'^(trả lời|câu trả lời|đáp án|trợ lý trả lời)[:.]?\s*',
                r'^(dựa vào|theo|căn cứ)\s+.*?[:]\s*',
                r'^(tôi|em)\s+',
            ]
            for pattern in clean_patterns:
                answer = re.sub(pattern, '', answer, flags=re.IGNORECASE)
            # Sanitize model echo of CONTEXT/QUESTION or exact question
            answer = _sanitize_model_output(answer, question, context)
            answer = answer.strip()

        return answer or "Không thể tạo câu trả lời từ thông tin đã cho"
    except Exception as e:
        return f"Lỗi sinh câu trả lời: {str(e)}"


async def async_generate_answer(llm, question: str, context: str, max_new_tokens: int = 512, temperature: float = 0.3,
                                max_prompt_chars: int = 12000, verbosity: str = 'normal') -> str:
    """Run the synchronous simple_generate_answer helper in a worker thread.

    Ollama's REST API is blocking when accessed via requests; offloading it keeps the
    event loop responsive while still honoring the semaphore-based concurrency guard.
    """
    return await asyncio.to_thread(
        simple_generate_answer,
        llm,
        question,
        context,
        max_new_tokens,
        temperature,
        max_prompt_chars,
        verbosity,
    )


_GEN_FAILURE_TOKENS = (
    "không thể tạo câu trả lời",
    "không thể tạo câu trả lời từ thông tin đã cho",
    "llm không khả dụng",
    "lỗi sinh câu trả lời",
    "[llm error]",
    "lỗi: không có",
    "không có output mới",
)


def _is_generation_failure(text: Optional[str]) -> bool:
    if not isinstance(text, str):
        return True
    stripped = text.strip()
    if not stripped:
        return True
    lower = stripped.lower()
    for token in _GEN_FAILURE_TOKENS:
        if token in lower:
            return True
    return False


def _trim_context_for_retry(context: str, limit: int = 2000) -> str:
    if not isinstance(context, str):
        return ""
    if len(context) <= limit:
        return context
    return context[:limit] + "…"


async def _try_llm_generation(question: str,
                              context: str,
                              *,
                              detail_mode: str,
                              mem_chain: Any,
                              llm: Any,
                              args: SimpleNamespace,
                              sess_id: str) -> Dict[str, Any]:
    """Attempt to generate an answer with the configured LLM/memory chain.

    Returns a dict containing keys:
      - answer: str | None
      - error: Optional[str]
      - error_type: Optional[str] (queue_timeout | queue_acquire | missing | generation)
      - slot_acquired: bool
      - wait_duration: float (seconds)
      - strategy: Optional[str] (memory | llm)
      - attempt_details: List[Dict[str, Any]]
    """
    result: Dict[str, Any] = {
        "answer": None,
        "error": None,
        "error_type": None,
        "slot_acquired": False,
        "wait_duration": 0.0,
        "attempts": 0,
        "fallback_used": False,
        "strategy": None,
        "attempt_details": [],
        "mem_chain_error": None,
    }

    llm_semaphore = STATE.get("query_semaphore")
    queue_timeout = float(STATE.get("query_queue_timeout", 600.0) or 600.0)
    wait_start = time.perf_counter()
    release_handle = None

    if llm_semaphore:
        try:
            await asyncio.wait_for(llm_semaphore.acquire(), timeout=queue_timeout)
            result["slot_acquired"] = True
            release_handle = llm_semaphore
            result["wait_duration"] = time.perf_counter() - wait_start
        except asyncio.TimeoutError:
            result["error"] = "llm_queue_timeout"
            result["error_type"] = "queue_timeout"
            return result
        except Exception as exc:
            result["error"] = str(exc)
            result["error_type"] = "queue_acquire"
            # Proceed without semaphore slot; avoid blocking answer generation entirely
            release_handle = None
    try:
        total_attempts = 0
        if mem_chain:
            try:
                out = await asyncio.to_thread(
                    mem_chain.invoke,
                    {"question": question, "context": context},
                    config={"configurable": {"session_id": sess_id}},
                )
                answer = (getattr(out, "content", None) or str(out) or "").strip()
                total_attempts += 1
                result["attempts"] = total_attempts
                result["answer"] = answer
                result["strategy"] = "memory"
                result["attempt_details"].append({
                    "mode": "memory_chain",
                    "status": "ok" if not _is_generation_failure(answer) else "fallback",
                    "context_chars": len(context or ""),
                    "verbosity": detail_mode,
                })
                if not _is_generation_failure(answer):
                    return result
                result["fallback_used"] = bool(llm)
            except Exception as mem_exc:
                result["mem_chain_error"] = str(mem_exc)
                result["attempt_details"].append({
                    "mode": "memory_chain",
                    "status": "error",
                    "error": str(mem_exc),
                })
                result["fallback_used"] = bool(llm)
                if not llm:
                    result["error"] = str(mem_exc)
                    result["error_type"] = "generation"
                    return result

        if llm:
            base_temp = float(getattr(args, "prompt_temperature", 0.3) or 0.3)
            base_max_new = int(getattr(args, "llm_max_new", 512) or 512)
            trimmed_context = _trim_context_for_retry(context, limit=min(getattr(args, "ctx_max_chars", 16000), 2200))
            attempt_specs: List[Dict[str, Any]] = []
            attempt_specs.append({
                "verbosity": detail_mode,
                "temperature": base_temp,
                "max_new": base_max_new,
                "context": context,
            })
            if detail_mode != "minimal":
                attempt_specs.append({
                    "verbosity": "minimal",
                    "temperature": max(0.15, base_temp - 0.1),
                    "max_new": min(base_max_new, 768),
                    "context": trimmed_context,
                })
            # Final guard attempt: low temperature, shorter answer
            attempt_specs.append({
                "verbosity": "normal" if detail_mode == "detailed" else detail_mode,
                "temperature": min(base_temp, 0.25),
                "max_new": min(base_max_new, 640),
                "context": _trim_context_for_retry(trimmed_context, limit=1500),
            })

            answer_text: Optional[str] = None
            llm_attempt_used = 0
            for idx, spec in enumerate(attempt_specs, 1):
                try:
                    answer_text = await async_generate_answer(
                        llm,
                        question,
                        spec["context"],
                        max_new_tokens=spec["max_new"],
                        temperature=spec["temperature"],
                        verbosity=spec["verbosity"],
                    )
                    status = "ok"
                except Exception as exc:
                    answer_text = f"[LLM ERROR] {exc}"
                    status = "error"
                llm_attempt_used = idx
                result["attempt_details"].append({
                    "mode": "llm",
                    "status": status if not _is_generation_failure(answer_text) else "fallback",
                    "temperature": spec["temperature"],
                    "max_new_tokens": spec["max_new"],
                    "context_chars": len(spec["context"] or ""),
                })
                if not _is_generation_failure(answer_text):
                    break

            total_attempts += llm_attempt_used
            result["attempts"] = total_attempts or llm_attempt_used
            result["answer"] = (answer_text or "").strip()
            result["fallback_used"] = result.get("fallback_used", False) or llm_attempt_used > 1
            result["strategy"] = "llm"
        else:
            if result.get("error") is None:
                result["error"] = "llm_missing"
                result["error_type"] = "missing"
            return result
    except Exception as exc:
        result["error"] = str(exc)
        result["error_type"] = "generation"
    finally:
        if release_handle:
            try:
                release_handle.release()
            except Exception:
                pass
    return result


def _summarize_llm_result(result: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not result:
        return None
    summary: Dict[str, Any] = {
        "strategy": result.get("strategy"),
        "attempts": result.get("attempts"),
        "fallback_used": bool(result.get("fallback_used")),
        "queue_wait_ms": int(float(result.get("wait_duration") or 0.0) * 1000),
        "slot_acquired": bool(result.get("slot_acquired")),
        "attempt_details": result.get("attempt_details") or [],
    }
    if result.get("error"):
        summary["error"] = result.get("error")
    if result.get("error_type"):
        summary["error_type"] = result.get("error_type")
    if result.get("mem_chain_error"):
        summary["mem_chain_error"] = result.get("mem_chain_error")
    return summary

class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = "api"
    debug: Optional[bool] = False
    # debug_level: minimal | normal | detailed
    debug_level: Optional[str] = "normal"

class QueryResponse(BaseModel):
    # Simplified JSON-only structure (no terminal box drawing)
    info: Dict[str, Any] = {}
    question: Dict[str, Any] = {}
    signals: Dict[str, Any] = {}
    hits: Dict[str, Any] = {}
    context: Dict[str, Any] = {}
    answer: Dict[str, Any] = {}
    notice: Optional[str] = None
    # Extra details to expose what semantic and BM25 produced (previews, contexts, etc.)
    details: Dict[str, Any] = {}


class LoadLogResponse(BaseModel):
    file: str
    total: int
    offset: int
    limit: Optional[int]
    count: int
    items: List[Dict[str, Any]]
    summary: Optional[Dict[str, Any]] = None


class LoadTestRequest(BaseModel):
    url: str = Field(
        default="http://localhost:8011/query",
        description="Endpoint nhận POST để gửi câu hỏi.",
    )
    questions: str = Field(
        default="questions.csv",
        description="Đường dẫn tới file CSV câu hỏi (tương đối hoặc tuyệt đối trong workspace).",
    )
    total: int = Field(default=100, ge=1, le=10000, description="Tổng số request cần gửi")
    concurrency: int = Field(default=20, ge=1, le=500, description="Số request đồng thời tối đa")
    session_id: str = Field(default="load-test", description="Session ID gửi cùng mỗi request")
    http_timeout: float = Field(default=650.0, gt=0, description="Timeout mỗi request (giây)")
    max_retries: int = Field(default=0, ge=0, le=5, description="Số lần retry khi request thất bại")
    dry_run: bool = Field(default=False, description="Chỉ kiểm tra dữ liệu, không gửi request")
    save_responses: Optional[str] = Field(
        default="out/load.jsonl",
        description="Lưu kết quả dạng JSONL (mặc định out/load.jsonl). Để null nếu không lưu.",
    )
    return_results: bool = Field(
        default=True,
        description="Trả về toàn bộ danh sách kết quả trong response.",
    )


class LoadTestResponse(BaseModel):
    status: Literal["dry-run", "completed"]
    config: Dict[str, Any]
    batches: int
    duration_s: Optional[float] = None
    summary: Optional[Dict[str, Any]] = None
    results_path: Optional[str] = None
    results: Optional[List[Dict[str, Any]]] = None

app = FastAPI(title="AI.LLM Retriever API", version="1.0.0")

# Global state (initialized at startup)
STATE: Dict[str, Any] = {
    "args": None,
    "index": None,
    "records": None,
    "sem": None,
    "bm25_all": None,
    "qa_records": None,
    "bm25_qa": None,
    "llm": None,
    "mem_chain": None,
    # lightweight per-session conversational memory (recent turns window)
    "conv": {},
    "query_semaphore": None,
    "query_queue_timeout": 600.0,
}

# ---------- Logging setup ----------
# Create a dedicated logger for this server to make it easy to grep in logs
logger = logging.getLogger("ai.llm.retriever.api")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[AI.LLM] %(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)


def _build_default_args() -> SimpleNamespace:
    # Mirror defaults from main.py argparse
    args = SimpleNamespace(
        embed_root="embed_data",
        channels=["dialogue", "lesson"],
        qa600_path="data/qa600.json",
        qa600_topn=3,
        qa600_thr=0.9,
        model_id=os.getenv("EMB_MODEL", "AITeamVN/Vietnamese_Embedding"),
        device="cpu",
        topk=3,
        bm25_topn=3,
        bm25_k1=1.5,
        bm25_b=0.75,
        sem_limit=3,
        llm_off=False,
        llm_model_id=os.getenv("LLM_MODEL_ID", "phogpt4b-ft"),
        llm_temp=float(os.getenv("LLM_TEMP", "0.3")),
        llm_ollama_host=os.getenv("LLM_OLLAMA_HOST", "http://localhost:11434"),
        # Set high output limits for full answers
        llm_max_new=1536,
        llm_provider="ollama",
        llm_timeout_connect=int(os.getenv("LLM_TIMEOUT_CONNECT", "10")),
        llm_timeout_read=int(os.getenv("LLM_TIMEOUT_READ", "600")),
        llm_keep_alive=os.getenv("LLM_KEEP_ALIVE", "5m"),
        hf_device="cpu",
        session_id="api",
        memory_max_turns=4,
        conv_memory_depth=int(os.getenv("CONV_MEMORY_DEPTH", "4")),
        memory_max_chars=1200,
        enable_simple_prompting=True,
        prompt_temperature=0.3,
        enable_memory=os.getenv("LLM_ENABLE_MEMORY", "0") in ("1", "true", "True"),
        enable_answer_preface=True,
        ctx_topn=3,
        ctx_max_chars=16000,
        ctx_item_max_chars=2000,
        ctx_join_window=0,
        thr_warn=0.5,
        min_query_chars=4,
        min_keyword_len=2,
        no_print_prompt=False,
        max_preview_len=180,
        ollama_num_ctx=4096,
        ollama_num_thread=None,
        max_concurrent_queries=max(1, int(os.getenv("MAX_CONCURRENT_QUERIES", "4"))),
    queue_timeout_sec=max(0.1, float(os.getenv("QUERY_QUEUE_TIMEOUT", "600"))),
    )
    return args


@app.on_event("startup")
def startup_event():
    # 1) Args and path discovery
    args = _build_default_args()
    # Normalize stale Windows path on Linux: this is already handled inside main; here we rely on fallback

    # Resolve embed root and qa600 path
    args.embed_root = _resolve_embed_root(args.embed_root, args.channels)
    args.qa600_path = _maybe_resolve_qa600_path(args.qa600_path)

    # 2) Load embedded corpus and semantic
    index, records = load_corpus(root=args.embed_root, channels=tuple(args.channels))
    sem = SemanticSearcher(index, model_id=args.model_id, device=args.device)

    # 3) BM25 over ALL embedded records
    bm25_docs_all = []
    for r in records:
        q, a = _extract_q_and_a(r)
        bm25_docs_all.append("\n".join([x for x in [q, a] if x]) if (q or a) else _get_text(r))
    bm25_all = BM25(bm25_docs_all, k1=args.bm25_k1, b=args.bm25_b)

    # 3a) BM25 over dialogue-only records for lightweight QA suggestions
    dialogue_records, dialogue_docs = [], []
    for r in records:
        if (r.get("channel") or "").lower() != "dialogue":
            continue
        dialogue_records.append(r)
        q, a = _extract_q_and_a(r)
        doc = "\n".join([x for x in [q, a] if x]) if (q or a) else (_get_text(r) or "")
        dialogue_docs.append(doc)
    bm25_dialogue = BM25(dialogue_docs, k1=args.bm25_k1, b=args.bm25_b) if dialogue_docs else None

    # 4) Load qa600.json (BM25-only)
    qa_records, qa_docs = [], []
    if args.qa600_path and os.path.isfile(args.qa600_path):
        with open(args.qa600_path, "r", encoding="utf-8") as f:
            items = json.load(f)
        if isinstance(items, dict):
            items = [items]
        for it in items:
            q = (it.get("question") or "").strip()
            a = (it.get("answer") or it.get("answer_text") or "").strip()
            if not (q or a):
                continue
            qa_records.append({
                "text": (q + ("\n" + a if a else "")).strip(),
                "qa": {"question": q or None, "answer": a or None},
                "meta": {"order": it.get("order"), "importance": it.get("importance", False)},
                "file": os.path.basename(args.qa600_path),
                "channel": "qa600",
            })
            qa_docs.append("\n".join([x for x in [q, a] if x]))
    bm25_qa = BM25(qa_docs, k1=args.bm25_k1, b=args.bm25_b) if qa_docs else None

    # 5) LLM / Memory
    mem_chain = None
    llm = None
    if not args.llm_off:
        llm = build_llm_from_args(args)
        # Build lightweight HF-style memory chain only when memory is enabled and
        # the LLM object appears compatible (duck-typing: has .generate).
        if llm and getattr(args, 'enable_memory', False):
            try:
                if hasattr(llm, 'generate'):
                    mem_chain = build_memory_chain_hf(
                        llm,
                        max_turns=args.memory_max_turns,
                        max_chars=args.memory_max_chars
                    )
            except Exception:
                mem_chain = None

    STATE.update({
        "args": args,
        "index": index,
        "records": records,
        "sem": sem,
        "bm25_all": bm25_all,
        "dialogue_records": dialogue_records,
        "bm25_dialogue": bm25_dialogue,
        "qa_records": qa_records,
        "bm25_qa": bm25_qa,
        "llm": llm,
        "mem_chain": mem_chain,
    })
    # Configure request concurrency guard
    try:
        max_concurrent = max(1, int(getattr(args, "max_concurrent_queries", 4)))
    except Exception:
        max_concurrent = 4
    STATE["query_semaphore"] = asyncio.Semaphore(max_concurrent)
    try:
        queue_timeout = float(getattr(args, "queue_timeout_sec", 6.0))
    except Exception:
        queue_timeout = 6.0
    STATE["query_queue_timeout"] = max(0.1, queue_timeout)
    # Log which LLM will be used at startup
    try:
        logger.info(
            "Startup: llm_provider=%s llm_model_id=%s llm_off=%s mem_chain=%s",
            args.llm_provider, args.llm_model_id, args.llm_off, bool(mem_chain)
        )
        if llm is not None:
            logger.info(
                "LLM ready: class=%s host=%s temperature=%s max_new_tokens=%s",
                type(llm).__name__, getattr(llm, 'host', None), getattr(llm, 'temperature', None), getattr(llm, 'max_new_tokens', None)
            )
        else:
            logger.info("LLM is disabled or not initialized.")
        logger.info(
            "Concurrency guard: max_concurrent=%s timeout=%.2fs",
            max_concurrent,
            STATE.get("query_queue_timeout", 600.0)
        )
    except Exception:
        pass


def _get_text(r: Dict[str, Any]) -> str:
    # Mirror main._extract_text behavior
    for key in ["text", "content", "search_text", "body", "chunk_text", "paragraph"]:
        v = r.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    q = (r.get("qa") or {}).get("question") or r.get("question")
    a = (r.get("qa") or {}).get("answer") or r.get("answer")
    q = q.strip() if isinstance(q, str) and q.strip() else None
    a = a.strip() if isinstance(a, str) and a.strip() else None
    if q or a:
        return ("Q: " + (q or "") + (f"\nA: {a}" if a else ""))
    return ""


def _preview_hits(records: List[Dict[str, Any]], hits: List[Dict[str, Any]], max_preview_len: int = 180) -> List[Dict[str, Any]]:
    out = []
    for h in (hits or []):
        rid = h.get("rid")
        if rid is None or rid < 0 or rid >= len(records):
            continue
        r = records[rid]
        txt = _get_text(r) or ""
        if len(txt) > max_preview_len:
            txt = txt[:max_preview_len] + "…"
        out.append({
            "rid": rid,
            "score": round(float(h.get("score", 0.0)), 4),
            "channel": r.get("channel"),
            "file": r.get("file"),
            "preview": txt,
        })
    return out


def _sources_from_hits(records: List[Dict[str, Any]], hits: List[Dict[str, Any]], limit: int = 3) -> List[str]:
    """Build ordered, deduplicated list of human-readable sources from hit metas.
    Each source is typically formatted as "{subject_name} - Bài {lesson}".
    """
    seen, out = set(), []
    for h in (hits or [])[: max(1, limit)]:
        rid = h.get("rid")
        if rid is None or rid < 0 or rid >= len(records):
            continue
        meta = records[rid].get("meta") if isinstance(records[rid], dict) else None
        src = _build_source_from_meta(meta) if isinstance(meta, dict) else None
        if src and src not in seen:
            seen.add(src)
            out.append(src)
    return out


def _compute_confidence_and_label(answer_type: str,
                                  sem_max: Optional[float],
                                  evidence: Optional[float],
                                  qa_top_score: Optional[float],
                                  qa_thr: float) -> Dict[str, Any]:
    s = float(sem_max or 0.0)
    e = float(evidence or 0.0)
    q = float(qa_top_score or 0.0)
    conf = 0.0
    rationale = ""
    if answer_type == "qa600":
        sem_factor = min(1.0, max(0.0, s))
        qa_factor = q / max(qa_thr, 1e-6) if qa_thr else 1.0
        qa_factor = min(1.0, max(0.0, qa_factor))
        conf = max(0.6, min(1.0, 0.55 * sem_factor + 0.45 * qa_factor))
        rationale = f"sem_max={s:.2f}, qa_score={q:.2f} vs thr {qa_thr:.2f}"
    elif answer_type == "semantic":
        # Heavier weight on semantic similarity, some weight on direct-answer evidence
        conf = max(0.0, min(1.0, 0.6 * s + 0.4 * e))
        rationale = f"sem_max={s:.2f}, evidence={e:.2f}"
    elif answer_type in ("bm25-suggestions", "dialogue-suggestions"):
        # Suggestions only; inherently lower confidence
        conf = 0.35
        rationale = "BM25 gợi ý (không có căn cứ semantic đủ mạnh)"
    elif answer_type == "low-semantic-llm":
        # Low semantic score but LLM synthesis performed using available hints
        base = 0.45 * s + 0.35 * e + 0.15
        conf = max(0.25, min(0.65, base))
        rationale = f"low-semantic llm synthesis: sem_max={s:.2f}, evidence={e:.2f}"
    elif answer_type in ("no-evidence", "llm-off", "error"):
        conf = 0.2 if answer_type == "no-evidence" else 0.0
        rationale = answer_type
    else:
        conf = max(0.0, min(1.0, 0.5 * s + 0.3 * e))
        rationale = f"mixed: sem_max={s:.2f}, evidence={e:.2f}"

    if conf >= 0.75:
        label = "chắc chắn"
    elif conf >= 0.5:
        label = "phân vân"
    else:
        label = "không chắc chắn"

    return {"confidence": round(conf, 4), "certainty": label, "reason": rationale}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/load-results", response_model=LoadLogResponse)
async def get_load_results(
    path: str = Query("out/load.jsonl", description="Đường dẫn tương đối tới file JSONL, mặc định là out/load.jsonl"),
    offset: int = Query(0, ge=0, description="Số bản ghi bỏ qua"),
    limit: int = Query(50, ge=0, le=500, description="Số bản ghi tối đa trả về; 0 để lấy tất cả"),
    include_summary: bool = Query(True, description="Bao gồm thống kê tóm tắt"),
):
    effective_limit = None if limit == 0 else limit
    payload = _load_jsonl_records(
        path,
        offset=offset,
        limit=effective_limit,
        include_summary=include_summary,
    )
    return {
        "file": payload["path"],
        "total": payload["total"],
        "offset": offset,
        "limit": effective_limit,
        "count": len(payload["items"]),
        "items": payload["items"],
        "summary": payload.get("summary") if include_summary else None,
    }


@app.post("/load-test/async-run", response_model=LoadTestResponse)
async def load_test_async_run(req: LoadTestRequest):
    try:
        questions_path = _resolve_workspace_path(req.questions)
    except HTTPException as exc:
        raise exc

    try:
        questions = async_batch_runner.load_questions_csv(questions_path)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    batches = async_batch_runner.build_batches(questions, req.total)
    save_path = None
    if req.save_responses:
        save_path = _resolve_workspace_path(req.save_responses)

    config = {
        "url": req.url,
        "questions_file": str(questions_path),
        "questions_count": len(questions),
        "total": req.total,
        "concurrency": req.concurrency,
        "session_id": req.session_id,
        "http_timeout": req.http_timeout,
        "max_retries": req.max_retries,
        "save_responses": str(save_path) if save_path else None,
        "dry_run": req.dry_run,
    }

    if req.dry_run:
        return LoadTestResponse(
            status="dry-run",
            config=config,
            batches=len(batches),
            summary=None,
            duration_s=None,
            results_path=str(save_path) if save_path else None,
            results=None,
        )

    runner_args = _build_runner_namespace(req, questions_path, save_path)
    try:
        outcome = await async_batch_runner.run_load_test(runner_args, batches)
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"Thiếu dependency: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lỗi chạy load test: {exc}") from exc

    results = outcome.get("results", [])
    duration = outcome.get("duration")
    summary = async_batch_runner.summarise(results)

    if save_path:
        _write_jsonl(save_path, results)

    return LoadTestResponse(
        status="completed",
        config=config,
        batches=len(batches),
        duration_s=round(float(duration or 0.0), 2),
        summary=summary,
        results_path=str(save_path) if save_path else None,
        results=results if req.return_results else None,
    )


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="question is required")

    args = STATE["args"]
    records = STATE["records"]
    sem = STATE["sem"]
    bm25_all = STATE["bm25_all"]
    dialogue_records = STATE.get("dialogue_records") or []
    bm25_dialogue = STATE.get("bm25_dialogue")
    qa_records = STATE["qa_records"]
    bm25_qa = STATE["bm25_qa"]
    llm = STATE["llm"]
    mem_chain = STATE["mem_chain"]

    # 1) Preprocess input
    orig_q = (req.question or "").strip()
    cleaned_q = clean_user_input(orig_q)
    # Log incoming request and model info
    try:
        logger.info(
            "Request: sid=%s model=%s provider=%s debug=%s debug_level=%s q=%s",
            getattr(req, 'session_id', None) or getattr(args, 'session_id', None),
            args.llm_model_id,
            args.llm_provider,
            bool(getattr(req, 'debug', False)),
            getattr(req, 'debug_level', None),
            orig_q.replace('\n', ' ')[:200]
        )
    except Exception:
        pass

    # Small-talk / identity early handling (before short-length guard)
    if _is_smalltalk_identity(orig_q) or _is_smalltalk_identity(cleaned_q):
        return QueryResponse(
            info={"llm_provider": args.llm_provider, "llm_model_id": args.llm_model_id},
            question={"original": orig_q, "processed": cleaned_q, "resolved": cleaned_q},
            signals={"sem_max": None, "evidence": None, "confidence": 1.0},
            hits={},
            context={"display": "", "for_llm": ""},
            answer={"text": _HP_AI_REPLY, "type": "system", "certainty": "chắc chắn", "sources": []},
            details={"intent": "smalltalk_identity"},
        )

    if len(cleaned_q) < args.min_query_chars:
        return QueryResponse(
            info={"llm_provider": args.llm_provider, "llm_model_id": args.llm_model_id},
            question={"original": orig_q, "processed": cleaned_q, "resolved": cleaned_q},
            notice=f"Câu hỏi quá ngắn (< {args.min_query_chars} ký tự)."
        )

    # Session id for memory
    sess_id = _get_sess_id(req.session_id, args.session_id)

    if getattr(args, 'enable_memory', False):
        try:
            q_resolved = resolve_after_preprocess(cleaned_q, sess_id)
        except Exception:
            q_resolved = cleaned_q
    else:
        q_resolved = cleaned_q
    # Apply lightweight conversational resolution for short follow-ups
    q_resolved = _resolve_followup_with_memory(q_resolved, sess_id)
    # verbosity control: minimal | normal | detailed
    detail_mode = (getattr(req, 'debug_level', None) or 'normal').lower()
    q_proc = clean_text(q_resolved)
    if not q_proc or len(q_proc) < 2:
        return QueryResponse(
            info={"llm_provider": args.llm_provider, "llm_model_id": args.llm_model_id},
            question={"original": orig_q, "processed": cleaned_q, "resolved": q_resolved},
            notice="Câu hỏi không hợp lệ sau khi xử lý."
        )

    # Block offensive/abusive questions and return top-3 QA suggestions
    if _is_offensive_question(orig_q) or _is_offensive_question(cleaned_q) or _is_offensive_question(q_resolved):
        # Prefer dialogue-based suggestions to steer users back on topic
        if bm25_dialogue is not None:
            dialogue_suggest = bm25_dialogue.search(q_proc, topk=3, min_keyword_len=args.min_keyword_len)
            block = _build_bm25_only_block(dialogue_records, dialogue_suggest, topn=3, max_chars=700) if dialogue_suggest else ""
            sources = _sources_from_hits(dialogue_records, dialogue_suggest, limit=3) if dialogue_suggest else []
            assess = _compute_confidence_and_label("dialogue-suggestions", None, None, None, float(args.qa600_thr))
            resp = QueryResponse(
                info={"llm_provider": args.llm_provider, "llm_model_id": args.llm_model_id},
                question={"original": orig_q, "processed": cleaned_q, "resolved": q_resolved},
                signals={"confidence": assess["confidence"]},
                hits={"dialogue": dialogue_suggest or []},
                context={"display": block, "for_llm": ""},
                answer={"text": "Câu hỏi không hợp lệ", "type": "invalid", "certainty": assess["certainty"], "sources": sources},
                details={
                    "dialogue": {"top_hits": _preview_hits(dialogue_records, dialogue_suggest or [], args.max_preview_len)},
                    "reason": "blocked_offensive_language"
                },
                notice="Câu hỏi không hợp lệ (ngôn ngữ xúc phạm); hiển thị 3 gợi ý từ hội thoại."
            )
            try:
                logger.info(
                    "Answer: method=%s provider=%s model=%s overlap=%s grounded=%s",
                    'invalid-offensive', args.llm_provider, args.llm_model_id, 'n/a', False
                )
            except Exception:
                pass
            return resp
        # Fallback to BM25 over all records if qa600 is not available
        fallback_hits = bm25_all.search(q_proc, topk=3, min_keyword_len=args.min_keyword_len)
        block = _build_bm25_only_block(records, fallback_hits, topn=3, max_chars=700) if fallback_hits else ""
        sources = _sources_from_hits(records, fallback_hits, limit=3) if fallback_hits else []
        assess = _compute_confidence_and_label("bm25-suggestions", None, None, None, float(args.qa600_thr))
        resp = QueryResponse(
            info={"llm_provider": args.llm_provider, "llm_model_id": args.llm_model_id},
            question={"original": orig_q, "processed": cleaned_q, "resolved": q_resolved},
            signals={"confidence": assess["confidence"]},
            hits={"bm25": fallback_hits},
            context={"display": block, "for_llm": ""},
            answer={"text": "Câu hỏi không hợp lệ", "type": "invalid", "certainty": assess["certainty"], "sources": sources},
            details={
                "bm25": {"hits_previews": _preview_hits(records, fallback_hits, args.max_preview_len)},
                "reason": "blocked_offensive_language"
            },
            notice="Câu hỏi không hợp lệ (ngôn ngữ xúc phạm); hiển thị top 3 gợi ý BM25."
        )
        try:
            logger.info(
                "Answer: method=%s provider=%s model=%s overlap=%s grounded=%s",
                'invalid-offensive', args.llm_provider, args.llm_model_id, 'n/a', False
            )
        except Exception:
            pass
        return resp

    # 2) BM25 on qa600 first
    qa_hits = []
    if bm25_qa is not None:
        qa_hits = bm25_qa.search(q_proc, topk=max(3, args.qa600_topn), min_keyword_len=args.min_keyword_len)
        # Build a string body similar to _print_hits
        lines = []
        for i, h in enumerate(qa_hits, 1):
            r = qa_records[h["rid"]]
            score_show = h.get("score", h.get("score_raw", 0.0))
            lines.append(f"#{i:02d}  score={score_show:.2f}  [qa600/{r['file']}]")
            # Preview
            qa = r.get("qa") or {}
            parts = []
            if qa.get("question"): parts.append("Q: " + qa["question"])
            if qa.get("answer"): parts.append("A: " + qa["answer"])
            txt = " | ".join(parts)
            if len(txt) > args.max_preview_len: txt = txt[:args.max_preview_len] + "…"
            lines.append("     " + txt)
        # Keep qa_hits for later high-confidence shortcut once semantic score is available

    # 3) Semantic + BM25(all)
    se_hits = sem.search(q_proc, topk=args.topk)
    # Prioritize semantic hits coming from `lesson` channel first (align with main copy)
    try:
        lesson_hits = [h for h in se_hits if (records[h["rid"]].get("channel") == "lesson")]
        other_hits = [h for h in se_hits if (records[h["rid"]].get("channel") != "lesson")]
        se_hits = lesson_hits + other_hits
    except Exception:
        pass
    bm_hits = bm25_all.search(q_proc, topk=args.bm25_topn, min_keyword_len=args.min_keyword_len)
    blended_hits = _blend_semantic_bm25_hits(
        records,
        se_hits,
        bm_hits,
        question=q_proc,
        max_per_source=2,
    )
    mx = max([h["score"] for h in se_hits], default=0.0)
    if getattr(args, 'enable_memory', False):
        try:
            update_focus_after_sem(q_resolved, sess_id, sem_max=mx, threshold=args.thr_warn)
        except Exception:
            pass
    ev = _evidence_score(records, se_hits)

    # signals JSON
    signals = {"sem_max": round(float(mx), 4), "evidence": round(float(ev), 4)}

    # High-confidence shortcut: when semantic score is extremely high, surface qa600 answer directly
    if mx >= 0.9 and qa_hits:
        top = qa_hits[0]
        rec = qa_records[top["rid"]]
        ans_text_raw = (rec.get("qa") or {}).get("answer") or _get_text(rec) or ""
        if ans_text_raw.strip():
            top_score = float(top.get("score", top.get("score_raw", 0.0)))
            assess = _compute_confidence_and_label(
                answer_type="qa600",
                sem_max=mx,
                evidence=ev,
                qa_top_score=top_score,
                qa_thr=float(args.qa600_thr),
            )
            ans_text = _maybe_with_preface(ans_text_raw.strip(), orig_q, getattr(args, 'enable_answer_preface', True))
            ans_text = _strip_artifact_wrappers(ans_text)
            resp = QueryResponse(
                info={"llm_provider": args.llm_provider, "llm_model_id": args.llm_model_id},
                question={"original": orig_q, "processed": cleaned_q, "resolved": q_resolved},
                signals={**signals, "confidence": assess["confidence"]},
                hits={"semantic": se_hits, "bm25": bm_hits, "qa600": qa_hits, "blended": blended_hits},
                context={"display": "", "for_llm": ""},
                answer={"text": ans_text, "type": "qa600", "certainty": assess["certainty"], "sources": []},
                details={
                    "qa600": {
                        "top_hits": _preview_hits(qa_records, qa_hits, args.max_preview_len),
                        "threshold": args.qa600_thr,
                        "reason": assess["reason"],
                    },
                    "semantic": {
                        "hits_previews": _preview_hits(records, se_hits, args.max_preview_len),
                    }
                },
            )
            try:
                _update_conv_memory(sess_id, q_resolved, resp.answer.get("text", ""), orig_q)
            except Exception:
                pass
            try:
                logger.info(
                    "Answer: method=%s provider=%s model=%s overlap=%s grounded=%s",
                    'qa600', args.llm_provider, args.llm_model_id, 'n/a', False
                )
            except Exception:
                pass
            return resp

    # LLM off?
    if args.llm_off:
        assess = _compute_confidence_and_label("llm-off", mx, ev, None, float(args.qa600_thr))
        # Try to surface sources based on semantic hits even when LLM is off
        sem_sources = _sources_from_hits(records, blended_hits or se_hits, limit=args.ctx_topn or 3)
        return QueryResponse(
            info={"llm_provider": args.llm_provider, "llm_model_id": args.llm_model_id},
            question={"original": orig_q, "processed": cleaned_q, "resolved": q_resolved},
            signals={**signals, "confidence": assess["confidence"]},
            hits={"semantic": se_hits, "bm25": bm_hits, "blended": blended_hits},
            context={"display": "", "for_llm": ""},
            answer={"text": "", "type": "llm-off", "certainty": assess["certainty"], "sources": sem_sources},
            details={
                "semantic": {
                    "hits_previews": _preview_hits(records, se_hits, args.max_preview_len),
                    "context": "",
                },
                "bm25": {
                    "hits_previews": _preview_hits(records, bm_hits, args.max_preview_len),
                }
            }
        )

    # semantic >= 0.5 => chắc chắn
    if mx >= 0.5 and (mem_chain or llm):
        if blended_hits:
            ctx_seed = sort_by_definition_first(records, list(blended_hits))
        else:
            se_sorted = sorted(se_hits, key=lambda x: -x["score"])
            se_dedup = dedup_by_source(records, se_sorted, max_per_source=2)
            ctx_seed = sort_by_definition_first(records, se_dedup)
        ctx_ids = ctx_seed[:args.sem_limit]
        if args.ctx_join_window > 0:
            semantic_blocks = join_adjacent_chunks(records, ctx_ids, window=args.ctx_join_window, max_chars=args.ctx_item_max_chars)
        else:
            semantic_blocks = [(records[h["rid"]].get("text") or "")[: args.ctx_item_max_chars] for h in ctx_ids]
        main_context = _build_context_block_from_hits(semantic_blocks, args.ctx_topn, args.ctx_item_max_chars, args.ctx_max_chars)

        bm25_context = ""
        if bm_hits:
            hints = _build_bm25_hints_block(records, bm_hits[:2], max_chars_per_item=800)
            if hints.strip():
                bm25_context = f"\n\n[THAM KHẢO BM25]\n{hints}"

        display_context = main_context + bm25_context
        final_context_for_llm = main_context
        # Strip BM25 hint section for answer synthesis to avoid polluting bullets or extractions
        ans_context = _strip_bm25_hints(display_context)

        gen_result = await _try_llm_generation(
            q_resolved,
            final_context_for_llm,
            detail_mode=detail_mode,
            mem_chain=mem_chain,
            llm=llm,
            args=args,
            sess_id=sess_id,
        )
        llm_summary = _summarize_llm_result(gen_result)

        if gen_result.get("error_type") == "queue_timeout":
            assess_busy = _compute_confidence_and_label("error", mx, ev, None, float(args.qa600_thr))
            busy_details: Dict[str, Any] = {
                "reason": "llm_queue_timeout",
                "queue_timeout_sec": STATE.get("query_queue_timeout", 600.0),
            }
            if getattr(req, 'debug', False):
                perf = busy_details.setdefault("performance", {})
                perf.update({
                    "llm_queue_wait_ms": int(gen_result.get("wait_duration", 0.0) * 1000),
                    "concurrency_limit": getattr(args, "max_concurrent_queries", None),
                })
                if llm_summary:
                    perf.setdefault("llm_strategy", llm_summary.get("strategy"))
                    perf.setdefault("llm_attempt_details", llm_summary.get("attempt_details"))
            if llm_summary:
                busy_details["llm"] = llm_summary
            return QueryResponse(
                info={"llm_provider": args.llm_provider, "llm_model_id": args.llm_model_id},
                question={"original": orig_q, "processed": cleaned_q, "resolved": q_resolved},
                signals={**signals, "confidence": assess_busy["confidence"]},
                hits={"semantic": se_hits, "bm25": bm_hits, "qa600": qa_hits, "blended": blended_hits},
                context={"display": "", "for_llm": ""},
                answer={"text": "Hệ thống đang bận, vui lòng thử lại sau.", "type": "busy", "certainty": assess_busy["certainty"], "sources": []},
                details=busy_details,
                notice="Hệ thống đang bận, vui lòng thử lại sau ít phút."
            )

        answer = (gen_result.get("answer") or "").strip()
        llm_slot_acquired = bool(gen_result.get("slot_acquired"))
        llm_wait_duration = float(gen_result.get("wait_duration") or 0.0)
        llm_attempts = int(gen_result.get("attempts") or 0)
        llm_fallback_used = bool(gen_result.get("fallback_used"))

        # Append hints if runner crashed
        if isinstance(answer, str) and "runner process has terminated" in answer:
            answer += ("\n[HƯỚNG DẪN] Ollama runner đã dừng đột ngột. Thử:" \
                       "\n- Giảm --llm-max-new (vd: 64 hoặc 128)" \
                       "\n- Giảm --ollama-num-ctx (vd: 512)" \
                       "\n- Dùng model nhẹ hơn (vd: mistral, q3 quant)" \
                       "\n- Kiểm tra log: journalctl -u ollama -f hoặc ollama serve -v")

        assess = _compute_confidence_and_label("semantic", mx, ev, None, float(args.qa600_thr))
        # If answer is empty or generic failure, try extracting directly from context
        fallback_needed = False
        ans_norm = (answer or "").strip().lower()
        if not ans_norm or ans_norm in {
            "không có câu trả lời",
            "không thể tạo câu trả lời",
            "không thể tạo câu trả lời từ thông tin đã cho",
            "llm không khả dụng",
            "lỗi tokenize",
            "lỗi decode",
            "không có output mới"
        } or ans_norm.startswith("[llm error]"):
            ext = _extract_direct_answer_from_text(ans_context)
            if ext:
                answer = ext
                # Slightly reduce confidence if we bypassed LLM
                assess = _compute_confidence_and_label("bm25-direct", mx, ev, None, float(args.qa600_thr))

        # Surface sources from the semantic context windows used for LLM
        sem_sources = _sources_from_hits(records, ctx_ids, limit=args.ctx_topn or 3)
        # If user question is yes/no but the extracted answer is a descriptive sentence without an explicit yes/no,
        # prefer to lead with a clear 'Có'/'Không' according to context polarity, then add bullets.
        if _is_yes_no_question(orig_q):
            p_ctx_tmp = _polarity_from_text(ans_context)
            if p_ctx_tmp in ("prohibit", "allow") and not ans_norm.startswith("có") and not ans_norm.startswith("không"):
                base = "Không" if p_ctx_tmp == "prohibit" else "Có"
                support_lines = _format_points_for_question(
                    _extract_key_points(ans_context, max_points=3),
                    orig_q,
                    limit=3,
                )
                lead_lines = [base] + support_lines if support_lines else [base]
                if answer and answer.strip():
                    answer = "\n".join(lead_lines + [answer.strip()])
                else:
                    answer = "\n".join(lead_lines)
        # Sanitize, apply no Q/A echo guard, and optional grounding/contradiction fix
        raw_answer = answer
        answer = _sanitize_llm_output(answer)
        answer = _guard_no_qa_echo(answer, ans_context, orig_q)
        # Hallucination detection: if answer has low overlap with context and is not just Có/Không, fallback
        try:
            overlap = _content_overlap_ratio(answer, ans_context)
        except Exception:
            overlap = 0.0
        allow_reasoning = _marks_reasoned_sections(answer) or _marks_reasoned_sections(raw_answer)
        if not allow_reasoning and _should_allow_reasoned_gap(orig_q, answer, ans_context):
            augmented = _build_reasoned_gap_answer(answer, ans_context)
            if augmented:
                answer = augmented
                allow_reasoning = True
        overlap_floor = 0.45
        # If overlap is low, but context shows clear evidence (either direct answer or strong BM25 hints),
        # prefer to synthesize an answer from the context instead of returning generic 'Chưa đủ căn cứ'.
        if (not allow_reasoning) and (not _is_short_answer(answer)) and overlap < overlap_floor:
            pts_raw = _extract_key_points(ans_context, max_points=3)
            support_points = _select_supporting_points(pts_raw, orig_q, limit=3)
            # If we have a definitional sentence or direct ext, use that as lead
            def_lead = _extract_definition_from_context(ans_context) or _extract_direct_answer_from_text(ans_context) or _extract_best_qna_answer(ans_context, q_resolved)
            if def_lead:
                lines = [def_lead]
                lead_low = def_lead.lower()
                for p_text in support_points:
                    if not p_text:
                        continue
                    if p_text.lower() in lead_low:
                        continue
                    lines.append(f"- {p_text}")
                answer = "\n".join(lines)
            elif support_points:
                answer = "\n".join(f"- {p_text}" for p_text in support_points if p_text)
            else:
                answer = "Chưa đủ căn cứ từ nội dung tham khảo"
        if not answer:
            # Try a principled fallback for yes/no questions based on context polarity
            p_ctx = _polarity_from_text(ans_context)
            if _is_yes_no_question(orig_q) and p_ctx in ("prohibit", "allow"):
                base = "Không" if p_ctx == "prohibit" else "Có"
                support_lines = _format_points_for_question(
                    _extract_key_points(ans_context, max_points=3),
                    orig_q,
                    limit=3,
                )
                answer = "\n".join([base] + support_lines) if support_lines else base
            else:
                support_lines = _format_points_for_question(
                    _extract_key_points(ans_context, max_points=4),
                    orig_q,
                    limit=4,
                )
                answer = "\n".join(support_lines) if support_lines else "Chưa đủ căn cứ từ nội dung tham khảo"
        ground = _ground_if_contradiction(answer, ans_context, orig_q)
        if ground.get("grounded") and ground.get("text"):
            answer = ground["text"]
        # Enrich short or single-sentence answers using context
        # Enrich more aggressively to avoid overly short answers
        enrich_points = 8 if detail_mode == 'detailed' else 6
        enriched = _enrich_short_answer(answer.strip(), ans_context, max_points=enrich_points)
        # Deduplicate sentences, strip internal exam artifacts and remove modal hallucinations before adding preface
        dedup_enriched = _dedupe_sentences(enriched)
        dedup_enriched = _strip_internal_headers(dedup_enriched)
        dedup_enriched = _remove_incorrect_modal_transformations(dedup_enriched, ans_context)
        preface_on = bool(getattr(args, 'enable_answer_preface', True)) and not _should_disable_preface(orig_q)
        final_text = _maybe_with_preface(dedup_enriched, orig_q, preface_on)
        final_text = _remove_question_echo_lines(final_text, orig_q)
        final_text = _strip_artifact_wrappers(final_text)
        details_payload: Dict[str, Any] = {
            "semantic": {
                "hits_previews": _preview_hits(records, se_hits, args.max_preview_len),
                "context": main_context,
            },
            "bm25": {
                "hits_previews": _preview_hits(records, bm_hits, args.max_preview_len),
                "hints": bm25_context.strip().replace("[THAM KHẢO BM25]", "").strip(),
            }
        }
        if blended_hits:
            details_payload["blended"] = {
                "ranked": [
                    {
                        "rid": h.get("rid"),
                        "score": h.get("score"),
                        "semantic": h.get("semantic"),
                        "bm25": h.get("bm25"),
                        "reasons": h.get("reasons"),
                    }
                    for h in blended_hits
                ]
            }
        if llm_summary:
            details_payload["llm"] = llm_summary
        signals_payload: Dict[str, Any] = {**signals, "confidence": assess["confidence"]}
        if llm_summary or llm_attempts or llm_fallback_used:
            signals_payload["llm"] = {
                "attempts": llm_attempts,
                "fallback_used": llm_fallback_used,
                "strategy": llm_summary.get("strategy") if llm_summary else None,
                "queue_wait_ms": llm_summary.get("queue_wait_ms") if llm_summary else int(llm_wait_duration * 1000),
            }
        resp = QueryResponse(
            info={"llm_provider": args.llm_provider, "llm_model_id": args.llm_model_id},
            question={"original": orig_q, "processed": cleaned_q, "resolved": q_resolved},
            signals=signals_payload,
            hits={"semantic": se_hits, "bm25": bm_hits, "qa600": qa_hits, "blended": blended_hits},
            context={"display": display_context, "for_llm": final_context_for_llm},
            answer={"text": final_text, "type": "semantic", "certainty": assess["certainty"], "sources": sem_sources},
            details=details_payload
        )
        if getattr(req, 'debug', False):
            perf = resp.details.setdefault("performance", {})
            perf.update({
                "llm_queue_wait_ms": int(llm_wait_duration * 1000),
                "concurrency_limit": getattr(args, "max_concurrent_queries", None),
                "queue_timeout_sec": STATE.get("query_queue_timeout", 600.0),
                "acquired_llm_slot": llm_slot_acquired,
                "llm_attempts": llm_attempts,
                "llm_fallback_used": llm_fallback_used,
            })
            if llm_summary:
                perf.setdefault("llm_strategy", llm_summary.get("strategy"))
                perf.setdefault("llm_attempt_details", llm_summary.get("attempt_details"))
        if llm_slot_acquired and llm_wait_duration > 0:
            try:
                logger.debug(
                    "LLM queue wait %.3fs (limit=%s)",
                    llm_wait_duration,
                    getattr(args, "max_concurrent_queries", None)
                )
            except Exception:
                pass
        # Log answer generation method and model details
        try:
            logger.info(
                "Answer: method=%s provider=%s model=%s strategy=%s attempts=%s fallback=%s overlap=%.3f grounded=%s",
                "semantic-llm",
                args.llm_provider,
                args.llm_model_id,
                llm_summary.get("strategy") if llm_summary else None,
                llm_attempts,
                llm_fallback_used,
                float(overlap) if 'overlap' in locals() else float('nan'),
                bool(ground.get('grounded')) if 'ground' in locals() else False,
            )
        except Exception:
            pass
        try:
            _update_conv_memory(sess_id, q_resolved, final_text, orig_q)
        except Exception:
            pass
        if getattr(req, 'debug', False):
            try:
                last_prompt = getattr(llm, 'last_prompt', None) if llm else None
                last_output = getattr(llm, 'last_output', None) if llm else None
            except Exception:
                last_prompt, last_output = None, None
            # Capture llm params if available
            llm_params = None
            try:
                if llm:
                    llm_params = {
                        'temperature': getattr(llm, 'temperature', None),
                        'top_p': getattr(llm, 'top_p', None),
                        'max_new_tokens': getattr(llm, 'max_new_tokens', getattr(llm, 'max_new_tokens', None))
                    }
            except Exception:
                llm_params = None
            # Compute answer overlap for debug visibility
            try:
                ans_overlap = _content_overlap_ratio(final_text or "", ans_context or "")
            except Exception:
                ans_overlap = None
            resp.details["debug"] = {
                "provider": args.llm_provider,
                "generation_method": "semantic-llm",
                "llm_model_id": args.llm_model_id,
                "debug_level": getattr(req, 'debug_level', None),
                "llm_params": llm_params,
                "llm_prompt": last_prompt,
                "llm_raw_output": last_output,
                "answer_overlap": ans_overlap,
                "grounded": bool(ground.get("grounded")),
                "ground_reason": ground.get("reason"),
                "missing_numbers": ground.get("missing_numbers") or [],
                "resolved_question": q_resolved,
                "llm_attempts": llm_attempts,
                "llm_fallback_used": llm_fallback_used,
                "llm_strategy": llm_summary.get("strategy") if llm_summary else None,
                "llm_attempt_details": llm_summary.get("attempt_details") if llm_summary else [],
                "llm_error": llm_summary.get("error") if llm_summary else None,
            }
        return resp
    # semantic < 0.5
    dialogue_suggest: List[Dict[str, Any]] = []
    if bm25_dialogue is not None:
        try:
            dialogue_suggest = bm25_dialogue.search(q_proc, topk=3, min_keyword_len=args.min_keyword_len)
        except Exception:
            dialogue_suggest = []

    if dialogue_suggest:
        dialogue_block = _build_bm25_only_block(dialogue_records, dialogue_suggest, topn=3, max_chars=700)
        dialogue_sources = _sources_from_hits(dialogue_records, dialogue_suggest, limit=3)

        # Build supplemental context from weak semantic / BM25 hits (if any)
        sem_context = ""
        candidate_hits = list(blended_hits) if blended_hits else []
        if not candidate_hits and se_hits:
            candidate_hits = sorted(se_hits, key=lambda x: -x["score"])
        if candidate_hits:
            trimmed = dedup_by_source(records, candidate_hits, max_per_source=1)
            ctx_ids = trimmed[: args.sem_limit]
            if ctx_ids:
                if args.ctx_join_window > 0:
                    semantic_blocks = join_adjacent_chunks(records, ctx_ids, window=args.ctx_join_window, max_chars=args.ctx_item_max_chars)
                else:
                    semantic_blocks = [(records[h["rid"]].get("text") or "")[: args.ctx_item_max_chars] for h in ctx_ids]
                sem_context = _build_context_block_from_hits(semantic_blocks, args.ctx_topn, args.ctx_item_max_chars, args.ctx_max_chars)

        bm25_context = ""
        if bm_hits:
            bm25_context = _build_bm25_only_block(records, bm_hits, topn=3, max_chars=700)

        combined_context_parts = []
        for block in [sem_context, bm25_context, dialogue_block]:
            if block and block.strip():
                combined_context_parts.append(block.strip())
        combined_context_display = "\n\n".join(combined_context_parts).strip()
        final_context_for_llm = combined_context_display
        ans_context = _strip_bm25_hints(final_context_for_llm)

        llm_result: Optional[Dict[str, Any]] = None
        llm_summary: Optional[Dict[str, Any]] = None
        generated_answer: Optional[str] = None
        low_detail_mode = 'detailed' if detail_mode not in ('minimal', 'detailed') else detail_mode
        if final_context_for_llm and not getattr(args, 'llm_off', False) and (llm or mem_chain):
            llm_result = await _try_llm_generation(
                q_resolved,
                final_context_for_llm,
                detail_mode=low_detail_mode,
                mem_chain=mem_chain,
                llm=llm,
                args=args,
                sess_id=sess_id,
            )
            llm_summary = _summarize_llm_result(llm_result)
            if llm_result and llm_result.get("error_type") == "queue_timeout":
                assess_busy = _compute_confidence_and_label("error", mx, ev, None, float(args.qa600_thr))
                busy_details: Dict[str, Any] = {
                    "reason": "llm_queue_timeout",
                    "queue_timeout_sec": STATE.get("query_queue_timeout", 600.0),
                }
                if getattr(req, 'debug', False):
                    perf = busy_details.setdefault("performance", {})
                    perf.update({
                        "llm_queue_wait_ms": int((llm_result.get("wait_duration") or 0.0) * 1000),
                        "concurrency_limit": getattr(args, "max_concurrent_queries", None),
                        "acquired_llm_slot": bool(llm_result.get("slot_acquired")),
                        "llm_attempts": llm_result.get("attempts"),
                        "llm_fallback_used": bool(llm_result.get("fallback_used")),
                        "llm_strategy": llm_result.get("strategy"),
                    })
                    if llm_summary:
                        perf.setdefault("llm_attempt_details", llm_summary.get("attempt_details"))
                if llm_summary:
                    busy_details["llm"] = llm_summary
                return QueryResponse(
                    info={"llm_provider": args.llm_provider, "llm_model_id": args.llm_model_id},
                    question={"original": orig_q, "processed": cleaned_q, "resolved": q_resolved},
                    signals={**signals, "confidence": assess_busy["confidence"]},
                    hits={"semantic": se_hits, "bm25": bm_hits, "dialogue": dialogue_suggest, "qa600": qa_hits, "blended": blended_hits},
                    context={"display": "", "for_llm": ""},
                    answer={"text": "Hệ thống đang bận, vui lòng thử lại sau.", "type": "busy", "certainty": assess_busy["certainty"], "sources": []},
                    details=busy_details,
                    notice="Hệ thống đang bận, vui lòng thử lại sau ít phút.",
                )
            generated_answer = (llm_result or {}).get("answer") or None
            if generated_answer:
                generated_answer = generated_answer.strip()

        if generated_answer:
            answer = generated_answer
            # Sanitize and post-process similar to high-confidence path
            raw_answer = answer
            answer = _sanitize_llm_output(answer)
            answer = _guard_no_qa_echo(answer, ans_context, orig_q)
            try:
                overlap = _content_overlap_ratio(answer, ans_context)
            except Exception:
                overlap = 0.0
            if _is_yes_no_question(orig_q):
                p_ctx_tmp = _polarity_from_text(ans_context)
                if p_ctx_tmp in ("prohibit", "allow") and not answer.lower().startswith(("có", "không")):
                    base = "Không" if p_ctx_tmp == "prohibit" else "Có"
                    support_lines = _format_points_for_question(
                        _extract_key_points(ans_context, max_points=3),
                        orig_q,
                        limit=3,
                    )
                    lead = "\n".join([base] + support_lines) if support_lines else base
                    answer = lead + ("\n" + answer if answer else "")
            allow_reasoning = _marks_reasoned_sections(answer) or _marks_reasoned_sections(raw_answer)
            if not allow_reasoning and _should_allow_reasoned_gap(orig_q, answer, ans_context):
                augmented = _build_reasoned_gap_answer(answer, ans_context)
                if augmented:
                    answer = augmented
                    allow_reasoning = True
            overlap_floor = 0.45
            if (not allow_reasoning) and (not _is_short_answer(answer)) and overlap < overlap_floor:
                pts_raw = _extract_key_points(ans_context, max_points=6)
                support_points = _select_supporting_points(pts_raw, orig_q, limit=6)
                def_lead = _extract_definition_from_context(ans_context) or _extract_direct_answer_from_text(ans_context) or _extract_best_qna_answer(ans_context, q_resolved)
                if def_lead:
                    lines = [def_lead]
                    lead_low = def_lead.lower()
                    for p_text in support_points:
                        if not p_text:
                            continue
                        if p_text.lower() in lead_low:
                            continue
                        lines.append(f"- {p_text}")
                    answer = "\n".join(lines)
                elif support_points:
                    answer = "\n".join(f"- {p_text}" for p_text in support_points if p_text)
            if not answer:
                answer = _extract_direct_answer_from_text(ans_context) or "Chưa đủ căn cứ từ nội dung tham khảo"
            ground = _ground_if_contradiction(answer, ans_context, orig_q)
            if ground.get("grounded") and ground.get("text"):
                answer = ground["text"]

            enrich_points = 8 if low_detail_mode == 'detailed' else 6
            enriched = _enrich_short_answer(answer.strip(), ans_context, max_points=enrich_points)
            dedup_enriched = _dedupe_sentences(enriched)
            dedup_enriched = _strip_internal_headers(dedup_enriched)
            dedup_enriched = _remove_incorrect_modal_transformations(dedup_enriched, ans_context)
            preface_on = bool(getattr(args, 'enable_answer_preface', True)) and not _should_disable_preface(orig_q)
            final_text = _maybe_with_preface(dedup_enriched, orig_q, preface_on)
            final_text = _remove_question_echo_lines(final_text, orig_q)
            final_text = _strip_artifact_wrappers(final_text)

            assess = _compute_confidence_and_label("low-semantic-llm", mx, ev, None, float(args.qa600_thr))
            sem_sources = _sources_from_hits(records, blended_hits or se_hits, limit=args.ctx_topn or 3)
            merged_sources = []
            for src in (sem_sources or []) + (dialogue_sources or []):
                if src and src not in merged_sources:
                    merged_sources.append(src)
            details_payload = {
                "semantic": {
                    "hits_previews": _preview_hits(records, se_hits, args.max_preview_len),
                    "context": sem_context,
                },
                "bm25": {
                    "hits_previews": _preview_hits(records, bm_hits, args.max_preview_len),
                    "context": bm25_context,
                },
                "dialogue": {
                    "top_hits": _preview_hits(dialogue_records, dialogue_suggest, args.max_preview_len),
                    "context": dialogue_block,
                }
            }
            if blended_hits:
                details_payload["blended"] = {
                    "ranked": [
                        {
                            "rid": h.get("rid"),
                            "score": h.get("score"),
                            "semantic": h.get("semantic"),
                            "bm25": h.get("bm25"),
                            "reasons": h.get("reasons"),
                        }
                        for h in blended_hits
                    ]
                }
            if llm_summary:
                details_payload["llm"] = llm_summary
            signals_payload = {**signals, "confidence": assess["confidence"]}
            if llm_summary or llm_result:
                attempts_val = llm_result.get("attempts") if llm_result else llm_summary.get("attempts")
                fallback_flag = bool(llm_result.get("fallback_used")) if llm_result else bool(llm_summary.get("fallback_used"))
                signals_payload["llm"] = {
                    "attempts": attempts_val,
                    "fallback_used": fallback_flag,
                    "strategy": llm_summary.get("strategy") if llm_summary else None,
                    "queue_wait_ms": llm_summary.get("queue_wait_ms") if llm_summary else int((llm_result.get("wait_duration") or 0.0) * 1000),
                }

            resp = QueryResponse(
                info={"llm_provider": args.llm_provider, "llm_model_id": args.llm_model_id},
                question={"original": orig_q, "processed": cleaned_q, "resolved": q_resolved},
                signals=signals_payload,
                hits={"semantic": se_hits, "bm25": bm_hits, "dialogue": dialogue_suggest, "qa600": qa_hits, "blended": blended_hits},
                context={"display": combined_context_display or dialogue_block, "for_llm": final_context_for_llm},
                answer={"text": final_text, "type": "low-semantic-llm", "certainty": assess["certainty"], "sources": merged_sources},
                details=details_payload,
                notice="Điểm semantic thấp (< 0.5); tổng hợp câu trả lời từ hội thoại và các đoạn tham khảo gần nhất.",
            )

            if llm_result and llm_result.get("slot_acquired") and llm_result.get("wait_duration"):
                try:
                    logger.debug(
                        "Low-semantic LLM queue wait %.3fs",
                        float(llm_result["wait_duration"]),
                    )
                except Exception:
                    pass
            if getattr(req, 'debug', False):
                perf = resp.details.setdefault("performance", {})
                if llm_result:
                    perf.update({
                        "llm_queue_wait_ms": int((llm_result.get("wait_duration") or 0.0) * 1000),
                        "acquired_llm_slot": bool(llm_result.get("slot_acquired")),
                        "debug_level": getattr(req, 'debug_level', None),
                        "generation_mode": "low-semantic",
                        "llm_attempts": llm_result.get("attempts"),
                        "llm_fallback_used": bool(llm_result.get("fallback_used")),
                    })
                    if llm_summary:
                        perf.setdefault("llm_strategy", llm_summary.get("strategy"))
                        perf.setdefault("llm_attempt_details", llm_summary.get("attempt_details"))
            try:
                logger.info(
                    "Answer: method=%s provider=%s model=%s strategy=%s fallback=%s overlap=%s grounded=%s",
                    'low-semantic-llm',
                    args.llm_provider,
                    args.llm_model_id,
                    llm_summary.get("strategy") if llm_summary else None,
                    bool(llm_result.get("fallback_used")) if llm_result else False,
                    float(overlap) if 'overlap' in locals() else float('nan'),
                    bool(ground.get('grounded')) if 'ground' in locals() else False,
                )
            except Exception:
                pass
            try:
                _update_conv_memory(sess_id, q_resolved, final_text, orig_q)
            except Exception:
                pass
            return resp

        # Default: fall back to suggestion listing (original behavior)
        preface_on = bool(getattr(args, 'enable_answer_preface', True)) and not _should_disable_preface(orig_q)
        fallback_display = combined_context_display or dialogue_block
        answer_text = _maybe_with_preface(dialogue_block, orig_q, preface_on)
        answer_text = _remove_question_echo_lines(answer_text, orig_q)
        answer_text = _strip_artifact_wrappers(answer_text)
        assess = _compute_confidence_and_label("dialogue-suggestions", mx, ev, None, float(args.qa600_thr))
        signals_payload = {**signals, "confidence": assess["confidence"]}
        if llm_summary:
            signals_payload["llm"] = {
                "attempts": llm_summary.get("attempts"),
                "fallback_used": llm_summary.get("fallback_used"),
                "strategy": llm_summary.get("strategy"),
                "queue_wait_ms": llm_summary.get("queue_wait_ms"),
                "error": llm_summary.get("error"),
                "error_type": llm_summary.get("error_type"),
            }
        resp = QueryResponse(
            info={"llm_provider": args.llm_provider, "llm_model_id": args.llm_model_id},
            question={"original": orig_q, "processed": cleaned_q, "resolved": q_resolved},
            signals=signals_payload,
            hits={"semantic": se_hits, "bm25": bm_hits, "dialogue": dialogue_suggest, "qa600": qa_hits, "blended": blended_hits},
            context={"display": fallback_display, "for_llm": ""},
            answer={"text": answer_text, "type": "dialogue-suggestions", "certainty": assess["certainty"], "sources": dialogue_sources},
            details={
                "semantic": {"hits_previews": _preview_hits(records, se_hits, args.max_preview_len)},
                "dialogue": {"top_hits": _preview_hits(dialogue_records, dialogue_suggest, args.max_preview_len)}
            },
            notice="Điểm semantic thấp (< 0.5); hiển thị 3 hội thoại gợi ý.",
        )
        if llm_summary:
            resp.details["llm"] = llm_summary
        try:
            logger.info(
                "Answer: method=%s provider=%s model=%s overlap=%s grounded=%s",
                'dialogue-suggestions', args.llm_provider, args.llm_model_id, 'n/a', False
            )
        except Exception:
            pass
        return resp

    if not bm_hits:
        # Do NOT forward to the LLM when there is no BM25 evidence and không có QA gợi ý.
        # Fallback: try BM25 over all records to surface suggestions
        fallback_hits = bm25_all.search(q_proc, topk=3, min_keyword_len=args.min_keyword_len)
        if fallback_hits:
            bm25_block = _build_bm25_only_block(records, fallback_hits, topn=3, max_chars=700)
            sem_context = ""
            candidate_hits = list(blended_hits) if blended_hits else []
            if not candidate_hits and se_hits:
                candidate_hits = sorted(se_hits, key=lambda x: -x["score"])
            if candidate_hits:
                trimmed = dedup_by_source(records, candidate_hits, max_per_source=1)
                ctx_ids = trimmed[: args.sem_limit]
                if ctx_ids:
                    if args.ctx_join_window > 0:
                        semantic_blocks = join_adjacent_chunks(records, ctx_ids, window=args.ctx_join_window, max_chars=args.ctx_item_max_chars)
                    else:
                        semantic_blocks = [(records[h["rid"]].get("text") or "")[: args.ctx_item_max_chars] for h in ctx_ids]
                    sem_context = _build_context_block_from_hits(semantic_blocks, args.ctx_topn, args.ctx_item_max_chars, args.ctx_max_chars)

            combined_context_parts = []
            for block in [sem_context, bm25_block]:
                if block and block.strip():
                    combined_context_parts.append(block.strip())
            combined_context_display = "\n\n".join(combined_context_parts).strip()
            final_context_for_llm = combined_context_display
            ans_context = _strip_bm25_hints(final_context_for_llm)

            llm_result = None
            llm_summary = None
            generated_answer = None
            low_detail_mode = 'detailed' if detail_mode not in ('minimal', 'detailed') else detail_mode
            if final_context_for_llm and not getattr(args, 'llm_off', False) and (llm or mem_chain):
                llm_result = await _try_llm_generation(
                    q_resolved,
                    final_context_for_llm,
                    detail_mode=low_detail_mode,
                    mem_chain=mem_chain,
                    llm=llm,
                    args=args,
                    sess_id=sess_id,
                )
                llm_summary = _summarize_llm_result(llm_result)
                if llm_result and llm_result.get("error_type") == "queue_timeout":
                    assess_busy = _compute_confidence_and_label("error", mx, ev, None, float(args.qa600_thr))
                    busy_details = {
                        "reason": "llm_queue_timeout",
                        "queue_timeout_sec": STATE.get("query_queue_timeout", 600.0),
                    }
                    if getattr(req, 'debug', False):
                        perf = busy_details.setdefault("performance", {})
                        perf.update({
                            "llm_queue_wait_ms": int((llm_result.get("wait_duration") or 0.0) * 1000),
                            "concurrency_limit": getattr(args, "max_concurrent_queries", None),
                            "acquired_llm_slot": bool(llm_result.get("slot_acquired")),
                            "llm_attempts": llm_result.get("attempts"),
                            "llm_fallback_used": bool(llm_result.get("fallback_used")),
                            "llm_strategy": llm_result.get("strategy"),
                        })
                        if llm_summary:
                            perf.setdefault("llm_attempt_details", llm_summary.get("attempt_details"))
                    if llm_summary:
                        busy_details["llm"] = llm_summary
                    return QueryResponse(
                        info={"llm_provider": args.llm_provider, "llm_model_id": args.llm_model_id},
                        question={"original": orig_q, "processed": cleaned_q, "resolved": q_resolved},
                        signals={**signals, "confidence": assess_busy["confidence"]},
                        hits={"semantic": se_hits, "bm25": fallback_hits, "qa600": qa_hits, "blended": blended_hits},
                        context={"display": "", "for_llm": ""},
                        answer={"text": "Hệ thống đang bận, vui lòng thử lại sau.", "type": "busy", "certainty": assess_busy["certainty"], "sources": []},
                        details=busy_details,
                        notice="Hệ thống đang bận, vui lòng thử lại sau ít phút.",
                    )
                generated_answer = (llm_result or {}).get("answer") or None
                if generated_answer:
                    generated_answer = generated_answer.strip()

            if generated_answer:
                answer = generated_answer
                raw_answer = answer
                answer = _sanitize_llm_output(answer)
                answer = _guard_no_qa_echo(answer, ans_context, orig_q)
                try:
                    overlap = _content_overlap_ratio(answer, ans_context)
                except Exception:
                    overlap = 0.0
                if _is_yes_no_question(orig_q):
                    p_ctx_tmp = _polarity_from_text(ans_context)
                    if p_ctx_tmp in ("prohibit", "allow") and not answer.lower().startswith(("có", "không")):
                        base = "Không" if p_ctx_tmp == "prohibit" else "Có"
                        support_lines = _format_points_for_question(
                            _extract_key_points(ans_context, max_points=3),
                            orig_q,
                            limit=3,
                        )
                        lead = "\n".join([base] + support_lines) if support_lines else base
                        answer = lead + ("\n" + answer if answer else "")
                allow_reasoning = _marks_reasoned_sections(answer) or _marks_reasoned_sections(raw_answer)
                if not allow_reasoning and _should_allow_reasoned_gap(orig_q, answer, ans_context):
                    augmented = _build_reasoned_gap_answer(answer, ans_context)
                    if augmented:
                        answer = augmented
                        allow_reasoning = True
                overlap_floor = 0.45
                if (not allow_reasoning) and (not _is_short_answer(answer)) and overlap < overlap_floor:
                    pts_raw = _extract_key_points(ans_context, max_points=6)
                    support_points = _select_supporting_points(pts_raw, orig_q, limit=6)
                    def_lead = _extract_definition_from_context(ans_context) or _extract_direct_answer_from_text(ans_context) or _extract_best_qna_answer(ans_context, q_resolved)
                    if def_lead:
                        lines = [def_lead]
                        lead_low = def_lead.lower()
                        for p_text in support_points:
                            if not p_text:
                                continue
                            if p_text.lower() in lead_low:
                                continue
                            lines.append(f"- {p_text}")
                        answer = "\n".join(lines)
                    elif support_points:
                        answer = "\n".join(f"- {p_text}" for p_text in support_points if p_text)
                if not answer:
                    answer = _extract_direct_answer_from_text(ans_context) or "Chưa đủ căn cứ từ nội dung tham khảo"
                ground = _ground_if_contradiction(answer, ans_context, orig_q)
                if ground.get("grounded") and ground.get("text"):
                    answer = ground["text"]

                enrich_points = 8 if low_detail_mode == 'detailed' else 6
                enriched = _enrich_short_answer(answer.strip(), ans_context, max_points=enrich_points)
                dedup_enriched = _dedupe_sentences(enriched)
                dedup_enriched = _strip_internal_headers(dedup_enriched)
                dedup_enriched = _remove_incorrect_modal_transformations(dedup_enriched, ans_context)
                preface_on = bool(getattr(args, 'enable_answer_preface', True)) and not _should_disable_preface(orig_q)
                final_text = _maybe_with_preface(dedup_enriched, orig_q, preface_on)
                final_text = _remove_question_echo_lines(final_text, orig_q)
                final_text = _strip_artifact_wrappers(final_text)

                assess = _compute_confidence_and_label("low-semantic-llm", mx, ev, None, float(args.qa600_thr))
                sem_sources = _sources_from_hits(records, blended_hits or se_hits, limit=args.ctx_topn or 3)
                bm_sources = _sources_from_hits(records, fallback_hits, limit=3)
                merged_sources = []
                for src in (sem_sources or []) + (bm_sources or []):
                    if src and src not in merged_sources:
                        merged_sources.append(src)

                details_payload = {
                    "semantic": {
                        "hits_previews": _preview_hits(records, se_hits, args.max_preview_len),
                        "context": sem_context,
                    },
                    "bm25": {
                        "hits_previews": _preview_hits(records, fallback_hits, args.max_preview_len),
                        "context": bm25_block,
                    },
                }
                if blended_hits:
                    details_payload["blended"] = {
                        "ranked": [
                            {
                                "rid": h.get("rid"),
                                "score": h.get("score"),
                                "semantic": h.get("semantic"),
                                "bm25": h.get("bm25"),
                                "reasons": h.get("reasons"),
                            }
                            for h in blended_hits
                        ]
                    }
                if llm_summary:
                    details_payload["llm"] = llm_summary
                signals_payload = {**signals, "confidence": assess["confidence"]}
                if llm_summary or llm_result:
                    attempts_val = llm_result.get("attempts") if llm_result else llm_summary.get("attempts")
                    fallback_flag = bool(llm_result.get("fallback_used")) if llm_result else bool(llm_summary.get("fallback_used"))
                    signals_payload["llm"] = {
                        "attempts": attempts_val,
                        "fallback_used": fallback_flag,
                        "strategy": llm_summary.get("strategy") if llm_summary else None,
                        "queue_wait_ms": llm_summary.get("queue_wait_ms") if llm_summary else int((llm_result.get("wait_duration") or 0.0) * 1000),
                    }

                resp = QueryResponse(
                    info={"llm_provider": args.llm_provider, "llm_model_id": args.llm_model_id},
                    question={"original": orig_q, "processed": cleaned_q, "resolved": q_resolved},
                    signals=signals_payload,
                    hits={"semantic": se_hits, "bm25": fallback_hits, "qa600": qa_hits, "blended": blended_hits},
                    context={"display": combined_context_display or bm25_block, "for_llm": final_context_for_llm},
                    answer={"text": final_text, "type": "low-semantic-llm", "certainty": assess["certainty"], "sources": merged_sources},
                    details=details_payload,
                    notice="Không tìm thấy căn cứ semantic đủ mạnh; tổng hợp câu trả lời từ các đoạn BM25 tương đồng nhất.",
                )
                if getattr(req, 'debug', False):
                    perf = resp.details.setdefault("performance", {})
                    if llm_result:
                        perf.update({
                            "llm_queue_wait_ms": int((llm_result.get("wait_duration") or 0.0) * 1000),
                            "acquired_llm_slot": bool(llm_result.get("slot_acquired")),
                            "generation_mode": "low-semantic",
                            "debug_level": getattr(req, 'debug_level', None),
                            "llm_attempts": llm_result.get("attempts"),
                            "llm_fallback_used": bool(llm_result.get("fallback_used")),
                        })
                        if llm_summary:
                            perf.setdefault("llm_strategy", llm_summary.get("strategy"))
                            perf.setdefault("llm_attempt_details", llm_summary.get("attempt_details"))
                try:
                    logger.info(
                        "Answer: method=%s provider=%s model=%s strategy=%s fallback=%s overlap=%s grounded=%s",
                        'low-semantic-llm',
                        args.llm_provider,
                        args.llm_model_id,
                        llm_summary.get("strategy") if llm_summary else None,
                        bool(llm_result.get("fallback_used")) if llm_result else False,
                        float(overlap) if 'overlap' in locals() else float('nan'),
                        bool(ground.get('grounded')) if 'ground' in locals() else False,
                    )
                except Exception:
                    pass
                try:
                    _update_conv_memory(sess_id, q_resolved, final_text, orig_q)
                except Exception:
                    pass
                return resp

            sources = _sources_from_hits(records, fallback_hits, limit=3)
            assess = _compute_confidence_and_label("bm25-suggestions", mx, ev, None, float(args.qa600_thr))
            preface_on = bool(getattr(args, 'enable_answer_preface', True)) and not _should_disable_preface(orig_q)
            answer_text = _maybe_with_preface(bm25_block, orig_q, preface_on)
            answer_text = _remove_question_echo_lines(answer_text, orig_q)
            answer_text = _strip_artifact_wrappers(answer_text)
            signals_payload = {**signals, "confidence": assess["confidence"]}
            if llm_summary or llm_result:
                attempts_val = llm_result.get("attempts") if llm_result else llm_summary.get("attempts")
                fallback_flag = bool(llm_result.get("fallback_used")) if llm_result else bool(llm_summary.get("fallback_used"))
                signals_payload["llm"] = {
                    "attempts": attempts_val,
                    "fallback_used": fallback_flag,
                    "strategy": llm_summary.get("strategy") if llm_summary else None,
                    "queue_wait_ms": llm_summary.get("queue_wait_ms") if llm_summary else int((llm_result.get("wait_duration") or 0.0) * 1000),
                    "error": llm_summary.get("error") if llm_summary else (llm_result.get("error") if llm_result else None),
                    "error_type": llm_summary.get("error_type") if llm_summary else (llm_result.get("error_type") if llm_result else None),
                }
            resp = QueryResponse(
                info={"llm_provider": args.llm_provider, "llm_model_id": args.llm_model_id},
                question={"original": orig_q, "processed": cleaned_q, "resolved": q_resolved},
                signals=signals_payload,
                hits={"semantic": se_hits, "bm25": fallback_hits, "qa600": qa_hits, "blended": blended_hits},
                context={"display": combined_context_display or bm25_block, "for_llm": ""},
                answer={"text": answer_text, "type": "bm25-suggestions", "certainty": assess["certainty"], "sources": sources},
                details={
                    "semantic": {"hits_previews": _preview_hits(records, se_hits, args.max_preview_len)},
                    "bm25": {"hits_previews": _preview_hits(records, fallback_hits, args.max_preview_len)}
                },
                notice="Không tìm thấy căn cứ semantic đủ mạnh; hiển thị các gợi ý BM25.",
            )
            if llm_summary:
                resp.details["llm"] = llm_summary
            try:
                logger.info(
                    "Answer: method=%s provider=%s model=%s overlap=%s grounded=%s",
                    'bm25-suggestions', args.llm_provider, args.llm_model_id, 'n/a', False
                )
            except Exception:
                pass
            return resp

        # No suggestions available at all
        assess = _compute_confidence_and_label("no-evidence", mx, ev, None, float(args.qa600_thr))
        resp = QueryResponse(
            info={"llm_provider": args.llm_provider, "llm_model_id": args.llm_model_id},
            question={"original": orig_q, "processed": cleaned_q, "resolved": q_resolved},
            signals={**signals, "confidence": assess["confidence"]},
            hits={"semantic": se_hits, "bm25": bm_hits, "qa600": qa_hits, "blended": blended_hits},
            context={"display": "", "for_llm": ""},
            answer={"text": "", "type": "no-evidence", "certainty": assess["certainty"], "sources": []},
            details={
                "semantic": {"hits_previews": _preview_hits(records, se_hits, args.max_preview_len)},
                "bm25": {"hits_previews": _preview_hits(records, bm_hits, args.max_preview_len)}
            },
            notice="Không tìm thấy căn cứ đủ mạnh và không có gợi ý từ BM25 hoặc QA600."
        )
        try:
            logger.info(
                "Answer: method=%s provider=%s model=%s overlap=%s grounded=%s",
                'no-evidence', args.llm_provider, args.llm_model_id, 'n/a', False
            )
        except Exception:
            pass
        return resp

    # Build BM25 suggestion block (prefer dialogue channel)
    try:
        bm_dialogue_hits = [h for h in bm_hits if records[h["rid"]].get("channel") == "dialogue"]
    except Exception:
        bm_dialogue_hits = []
    # Filter BM25 hits for basic relevance to the question to avoid off-topic suggestions
    def _tok_simple(s: str) -> List[str]:
        return [t for t in re.findall(r"[a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹ0-9]+", (s or "").lower()) if len(t) >= 2]
    q_tokens = set(_tok_simple(q_resolved))
    chosen_bm_hits = bm_dialogue_hits if bm_dialogue_hits else bm_hits
    filtered_hits = []
    if chosen_bm_hits:
        for h in chosen_bm_hits:
            rid = h.get("rid")
            if rid is None or rid < 0 or rid >= len(records):
                continue
            txt = _get_text(records[rid]) or ""
            tset = set(_tok_simple(txt))
            overlap = len(q_tokens & tset)
            # Require at least 2 overlapping tokens to keep this hit
            if overlap >= 2:
                filtered_hits.append(h)
    # If nothing passes the filter, keep original hits to avoid empty UI, but mark later as no-evidence if block is empty
    eff_hits = filtered_hits if filtered_hits else []
    bm25_block = _build_bm25_only_block(records, eff_hits, topn=3, max_chars=700)
    bm_sources = _sources_from_hits(records, eff_hits, limit=3)

    # Try extracting a direct short answer from BM25 block
    # If the question is 'bao nhiêu/mấy', try to extract numeric counts first
    extracted = _extract_counts_answer_from_text(bm25_block, q_resolved)
    if not extracted:
        extracted = _extract_direct_answer_from_text(bm25_block)

    if extracted:
        # Confidence based on BM25 top score
        top_bm = float(chosen_bm_hits[0].get("score", chosen_bm_hits[0].get("score_raw", 0.0)) if chosen_bm_hits else 0.0)
        if top_bm > 1.0:
            top_bm = min(1.0, top_bm / 40.0)
        conf = max(0.5, min(0.85, 0.45 + 0.5 * top_bm))
        label = "chắc chắn" if conf >= 0.75 else ("phân vân" if conf >= 0.5 else "không chắc chắn")
        enrich_points = 8 if detail_mode == 'detailed' else 6
        enriched = _enrich_short_answer(extracted, bm25_block, max_points=enrich_points)
        preface_on = bool(getattr(args, 'enable_answer_preface', True)) and not _should_disable_preface(orig_q)
        final_text = _maybe_with_preface(enriched, orig_q, preface_on)
        final_text = _remove_question_echo_lines(final_text, orig_q)
        final_text = _strip_artifact_wrappers(final_text)
        resp = QueryResponse(
            info={"llm_provider": args.llm_provider, "llm_model_id": args.llm_model_id},
            question={"original": orig_q, "processed": cleaned_q, "resolved": q_resolved},
            signals={**signals, "confidence": round(conf, 4)},
            hits={"semantic": se_hits, "bm25": bm_hits, "qa600": qa_hits, "blended": blended_hits},
            context={"display": bm25_block or "", "for_llm": ""},
            answer={"text": final_text,
                    "type": "bm25-direct", "certainty": label, "sources": bm_sources},
            details={
                "semantic": {"hits_previews": _preview_hits(records, se_hits, args.max_preview_len)},
                "bm25": {"hits_previews": _preview_hits(records, chosen_bm_hits, args.max_preview_len), "block": bm25_block}
            }
        )
        try:
            _update_conv_memory(sess_id, q_resolved, resp.answer.get("text", ""), orig_q)
        except Exception:
            pass
        return resp

    # Otherwise return suggestions-only (no LLM synthesis) — mirror main.py behavior
    assess = _compute_confidence_and_label("bm25-suggestions", mx, ev, None, float(args.qa600_thr))
    enrich_points = 8 if detail_mode == 'detailed' else 6
    # If we can form a concise counts lead for 'bao nhiêu/mấy', use it as the lead for enrichment
    counts_lead = _extract_counts_answer_from_text(bm25_block, q_resolved)
    enriched = _enrich_short_answer(counts_lead or "", bm25_block, max_points=enrich_points)
    preface_on = bool(getattr(args, 'enable_answer_preface', True)) and not _should_disable_preface(orig_q)
    final_text = _maybe_with_preface(enriched, orig_q, preface_on)
    final_text = _remove_question_echo_lines(final_text, orig_q)
    final_text = _strip_artifact_wrappers(final_text)
    if not enriched.strip():
        # If filtering removed all content, fall back to no-evidence
        return QueryResponse(
            info={"llm_provider": args.llm_provider, "llm_model_id": args.llm_model_id},
            question={"original": orig_q, "processed": cleaned_q, "resolved": q_resolved},
            signals={**signals, "confidence": assess["confidence"]},
            hits={"semantic": se_hits, "bm25": bm_hits, "qa600": qa_hits, "blended": blended_hits},
            context={"display": "", "for_llm": ""},
            answer={"text": "", "type": "no-evidence", "certainty": assess["certainty"], "sources": []},
            details={
                "semantic": {"hits_previews": _preview_hits(records, se_hits, args.max_preview_len)},
                "bm25": {"hits_previews": _preview_hits(records, eff_hits, args.max_preview_len)}
            },
            notice="Không tìm thấy căn cứ đủ liên quan để gợi ý."
        )
    resp = QueryResponse(
        info={"llm_provider": args.llm_provider, "llm_model_id": args.llm_model_id},
        question={"original": orig_q, "processed": cleaned_q, "resolved": q_resolved},
        signals={**signals, "confidence": assess["confidence"]},
        hits={"semantic": se_hits, "bm25": bm_hits, "qa600": qa_hits, "blended": blended_hits},
        context={"display": bm25_block or "", "for_llm": ""},
        answer={"text": final_text,
                "type": "bm25-suggestions", "certainty": assess["certainty"], "sources": bm_sources},
        details={
            "semantic": {"hits_previews": _preview_hits(records, se_hits, args.max_preview_len)},
            "bm25": {"hits_previews": _preview_hits(records, chosen_bm_hits, args.max_preview_len), "block": bm25_block}
        }
    )
    try:
        _update_conv_memory(sess_id, q_resolved, resp.answer.get("text", ""), orig_q)
    except Exception:
        pass
    return resp


def _resolve_embed_root(preferred_root: str, channels: List[str]) -> str:
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        ai_llm_root = os.path.dirname(here)
        workspace_root = os.path.dirname(ai_llm_root)
        candidates = []
        if preferred_root:
            candidates.append(preferred_root)
        candidates.append(os.path.join(here, "embed_data"))
        candidates.append(os.path.join(ai_llm_root, "embed_data"))
        candidates.append(os.path.join(workspace_root, "AI.TEST", "embed_data"))
        chs = [c for c in (channels or []) if isinstance(c, str) and c]
        for cand in candidates:
            if not cand:
                continue
            if os.path.isdir(cand):
                if not chs:
                    return cand
                for ch in chs:
                    ch_dir = os.path.join(cand, ch)
                    if os.path.isdir(ch_dir):
                        try:
                            has_json = any(fn.endswith('.json') for fn in os.listdir(ch_dir))
                        except Exception:
                            has_json = False
                        if has_json:
                            return cand
        return preferred_root
    except Exception:
        return preferred_root

def _maybe_resolve_qa600_path(current_path: str) -> str:
    try:
        if current_path and os.path.isfile(current_path):
            return current_path
        here = os.path.dirname(os.path.abspath(__file__))
        ai_llm_root = os.path.dirname(here)
        workspace_root = os.path.dirname(ai_llm_root)
        candidates = [
            current_path,
            os.path.join(here, "data", "qa600.json"),
            os.path.join(ai_llm_root, "data", "qa600.json"),
            os.path.join(workspace_root, "AI.TEST", "data", "qa600.json"),
        ]
        for cand in candidates:
            if cand and os.path.isfile(cand):
                return cand
    except Exception:
        pass
    return current_path


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("retriever.api_server:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
