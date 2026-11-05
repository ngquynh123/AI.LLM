import sys, types, os

def _patch_multihead_attention():
    try:
        import torch.nn as nn, inspect
        if 'dtype' in inspect.signature(nn.MultiheadAttention.__init__).parameters:
            print("[INFO] MultiheadAttention already supports dtype parameter")
            return
        if not hasattr(nn.MultiheadAttention, "_original_init"):
            nn.MultiheadAttention._original_init = nn.MultiheadAttention.__init__
        def patched_init(self, *args, **kwargs):
            # Loại các tham số không hỗ trợ ở bản torch cũ
            for k in ["dtype", "bias", "add_bias_kv", "add_zero_attn"]:
                if k in kwargs:
                    kwargs.pop(k, None)
            return nn.MultiheadAttention._original_init(self, *args, **kwargs)
        nn.MultiheadAttention.__init__ = patched_init
        print("[INFO] Successfully patched MultiheadAttention")
    except Exception as e:
        print(f"[WARN] Could not patch MultiheadAttention: {e}")

def _patch_flash_attn():
    try:
        import flash_attn_triton  # noqa: F401
    except Exception:
        print("[INFO] Creating flash_attn_triton mock...")
        m = types.ModuleType("flash_attn_triton")
        m.HAS_FLASH_ATTN = False
        m.available = lambda: False
        sys.modules["flash_attn_triton"] = m

_patch_multihead_attention()
_patch_flash_attn()

# Force offline mode by default (có thể override bằng env nếu muốn)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# ================= Imports =================
import argparse, re, unicodedata, shutil, textwrap, json
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from load_data import load_corpus
from model_semantic import SemanticSearcher
from bm25 import BM25
from model_llm import build_llm_from_args, generate_answer
from chat_memory import resolve_after_preprocess, update_focus_after_sem
from preprocess import clean_user_input

# ===== Small-talk / identity detection and answer enrichment helpers =====
_HP_AI_REPLY = "Xin chào. Tôi là HP_AI, tôi có thể giúp gì cho bạn?"

_GREET_PAT = re.compile(
    r"\b(xin chao|xin chào|chao|chào|hello|hi|hey|alo|yo)\b",
    re.IGNORECASE,
)
_IDENT_PAT = re.compile(
    r"\b(ban la ai|bạn là ai|ban la gi|bạn là gì|may la ai|mày là ai|cau la ai|cậu là ai|ai vay|ai vậy|who are you|what are you|introduce yourself|gioi thieu|giới thiệu)\b",
    re.IGNORECASE,
)

_PREFACE_LEAD_PAT = re.compile(r"^(vâng|dạ|xin chào|chào|hello|hi)\b", re.IGNORECASE)

def _is_smalltalk_identity(q: str) -> bool:
    qn = clean_text(q or "")
    if not qn:
        return False
    return bool(_GREET_PAT.search(qn) or _IDENT_PAT.search(qn))

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
        return _build_answer_preface(orig_question)
    if _PREFACE_LEAD_PAT.match(body):
        return body
    pre = _build_answer_preface(orig_question)
    return (pre + "\n" + body).strip()

def _split_sentences(text: str) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    parts = re.split(r"(?<=[\.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p and p.strip()]

def _is_short_answer(ans: str) -> bool:
    if not isinstance(ans, str):
        return True
    s = ans.strip().lower().strip('.!?:')
    if not s:
        return True
    if s in {"có", "không", "đúng", "sai"}:
        return True
    return len(s) <= 8

def _extract_key_points(context_text: str, max_points: int = 3, max_len: int = 220) -> List[str]:
    sents = _split_sentences(context_text)
    if not sents:
        return []
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
        score = 0.0
        low = sent.lower()
        for w, kws in keywords:
            if any(kw in low for kw in kws):
                score += w
        if "hầm" in low and "vượt" in low:
            score += 2.0
        if score > 0:
            txt = sent.strip()
            if len(txt) > max_len:
                txt = txt[:max_len] + "…"
            scored.append((score, txt))
    if not scored:
        return [s[:max_len] + ("…" if len(s) > max_len else "") for s in sents[:max_points]]
    scored.sort(key=lambda x: -x[0])
    seen, out = set(), []
    for _, txt in scored:
        key = txt.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(txt)
        if len(out) >= max_points:
            break
    return out

def _enrich_short_answer(answer_text: str, context_text: str, max_points: int = 3) -> str:
    if not _is_short_answer(answer_text):
        return (answer_text or "").strip()
    bullets = _extract_key_points(context_text, max_points=max_points)
    if not bullets:
        return (answer_text or "").strip()
    base = (answer_text or "").strip()
    lines = [base] if base else []
    for b in bullets:
        lines.append(f"- {b}")
    return "\n".join(lines).strip()

# Loại bỏ khả năng model vô tình lặp lại câu hỏi hoặc nội dung tham khảo
def _sanitize_model_output(answer: str, question: str, context: str) -> str:
    if not isinstance(answer, str):
        return ""
    s = answer
    # Xóa các khối có thẻ chỉ dẫn nếu bị echo
    try:
        s = re.sub(r"(?is)<CONTEXT>.*?</CONTEXT>", "", s)
        s = re.sub(r"(?is)<QUESTION>.*?</QUESTION>", "", s)
        # Xóa các dòng tiêu đề nếu xuất hiện
        s = re.sub(r"(?im)^NỘI DUNG LÝ THUYẾT THAM KHẢO\s*:?.*$", "", s)
        s = re.sub(r"(?im)^CÂU HỎI\s*:?.*$", "", s)
        # Xóa dòng chứa nguyên văn câu hỏi nếu khớp hoàn toàn
        q = (question or "").strip()
        if len(q) >= 4:
            q_esc = re.escape(q)
            s = re.sub(r"(?im)^\s*" + q_esc + r"\s*$", "", s)
            s = re.sub(r"(?im)^\s*Q\s*[:：]\s*" + q_esc + r"\s*$", "", s)
        # Làm gọn khoảng trắng
        s = s.strip()
        s = re.sub(r"\n{3,}", "\n\n", s)
    except Exception:
        # nếu có lỗi regex, trả về bản gốc đã strip
        s = answer.strip()
    return s

# =========================
# Vietnamese Traffic Law Prompting System (Simple & Basic)
def build_simple_vietnamese_prompt(question: str, context: str, sample_answer: str = "") -> str:
    """Xây dựng prompt đơn giản cho lý thuyết lái xe Việt Nam"""
    
    # System prompt cơ bản
    system_prompt = """Bạn là trợ lý AI chuyên về LÝ THUYẾT LÁI XE Ô TÔ tại Việt Nam.

Nhiệm vụ:
- Nhận vào 2 tham số:
    1) CÂU HỎI của người dùng.
    2) NỘI DUNG LÝ THUYẾT THAM KHẢO (là đoạn text đã được xử lý, trích xuất từ tài liệu chuẩn).

Quy tắc quan trọng:
- Chỉ được dựa vào NỘI DUNG LÝ THUYẾT THAM KHẢO để trả lời.
- Dùng NỘI DUNG LÝ THUYẾT THAM KHẢO.
- Khi chắc chắn: trả lời đúng trọng tâm, rõ ràng; dùng gạch đầu dòng để làm nổi bật ý chính.
- Luôn luôn sinh câu trả lời, không được từ chối trả lời.
- Không bịa đặt, không lấy dữ liệu có năm 2008, không suy đoán ngoài nội dung đã cho.
- Văn phong: tiếng Việt, thân thiện, trung lập, dễ hiểu.
- Tuyệt đối KHÔNG lặp lại/hiển thị lại CÂU HỎI hoặc NỘI DUNG LÝ THUYẾT THAM KHẢO trong phần trả lời.
- Không trích lại nhãn/tiêu đề như "NỘI DUNG LÝ THUYẾT THAM KHẢO" hay "CÂU HỎI"; không đưa vào thẻ <CONTEXT> hay <QUESTION> trong câu trả lời.
- Đầu ra: chỉ câu trả lời (không thêm JSON, metadata hay cấu trúc kỹ thuật)."""

    # Làm sạch input
    clean_question = (question or "").strip()
    clean_context = (context or "").strip()  
    
    # Xây dựng prompt
    prompt_parts = [system_prompt]
    
    # Thêm nội dung tham khảo (được bao trong thẻ để dễ hướng dẫn và lọc nếu bị echo)
    if clean_context:
        prompt_parts.append(f"\nNỘI DUNG LÝ THUYẾT THAM KHẢO:\n<CONTEXT>\n{clean_context}\n</CONTEXT>")
    else:
        prompt_parts.append(f"\nNỘI DUNG LÝ THUYẾT THAM KHẢO:\n(Không có nội dung tham khảo)")
    
    # Thêm câu hỏi (bao trong thẻ để dễ hướng dẫn và lọc nếu bị echo)
    prompt_parts.append(f"\nCÂU HỎI:\n<QUESTION>\n{clean_question}\n</QUESTION>")
    return "\n".join(prompt_parts)


def simple_generate_answer(llm, question: str, context: str, max_new_tokens: int = 512, temperature: float = 0.3, max_prompt_chars: int = 12000) -> str:
    """Sinh câu trả lời với prompt đơn giản - FIXED"""
    try:
        if not hasattr(llm, 'generate'):
            return "LLM không khả dụng"
        
        print(f"[DEBUG] Sử dụng simple prompting system")
        
        # Xây dựng prompt đơn giản
        simple_prompt = build_simple_vietnamese_prompt(question, context)
        
        print(f"[DEBUG] Prompt length: {len(simple_prompt)} chars")
        
        # Truncate if too long (raise limit to allow longer context/answers)
        if len(simple_prompt) > max_prompt_chars:
            print(f"[WARN] Prompt quá dài (> {max_prompt_chars} chars), cắt ngắn...")
            # giữ nhiều context hơn để sinh câu trả lời dài hơn
            keep_ctx = max(1000, min(len(context), max_prompt_chars - 1500))
            context = (context[:keep_ctx] + "...") if len(context) > keep_ctx else context
            simple_prompt = build_simple_vietnamese_prompt(question, context)
        
        # Thiết lập parameters
        if hasattr(llm, 'set_params'):
            llm.set_params(
                temperature=temperature,
                max_new_tokens=max_new_tokens
            )
        
        # Sinh câu trả lời
        stop_tokens = ["<|endoftext|>", "</s>"]  # Simplified stop tokens
        
        print(f"[DEBUG] Calling LLM generate...")
        answer = llm.generate(simple_prompt, stop=stop_tokens)
        print(f"[DEBUG] LLM returned: '{answer[:100] if answer else 'None'}...'")
        
        # Làm sạch câu trả lời
        if answer:
            # Loại bỏ các từ thừa đầu câu
            clean_patterns = [
                r'^(trả lời|câu trả lời|đáp án|trợ lý trả lời)[:.]?\s*',
                r'^(dựa vào|theo|căn cứ)\s+.*?[:]\s*',
                r'^(tôi|em)\s+',
            ]
            
            for pattern in clean_patterns:
                answer = re.sub(pattern, '', answer, flags=re.IGNORECASE)
            
            answer = answer.strip()

            # Loại bỏ việc model vô tình echo lại QUESTION/CONTEXT hoặc tiêu đề
            answer = _sanitize_model_output(answer, question, context)
        
        # ✅ CRITICAL FIX: Always return
        return answer or "Không thể tạo câu trả lời từ thông tin đã cho"
        
    except Exception as e:
        print(f"[ERROR] Simple generation failed: {e}")
        import traceback
        traceback.print_exc()
        # ✅ CRITICAL FIX: Return error message
        return f"Lỗi sinh câu trả lời: {str(e)}"
# ================= Pretty printing =================
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

def _clean_answer(s: str) -> str:
    s = (s or "").strip()
    while "\n\n\n" in s: s = s.replace("\n\n\n", "\n\n")
    return s

def _preview_line(record: Dict[str, Any]) -> str:
    qa = record.get("qa") or {}; ch = record.get("channel")
    if ch in ("dialogue","qa600"):
        parts = []
        if qa.get("question"): parts.append("Q: " + qa["question"])
        if qa.get("answer"):   parts.append("A: " + qa["answer"])
        txt = " | ".join(parts)
    else:
        txt = (record.get("text") or "").strip().splitlines()
        txt = txt[0] if txt else ""
    return (txt[:180] + "…") if len(txt) > 180 else txt

def _print_hits(title: str, records: List[Dict[str, Any]], hits: List[Dict[str, Any]]):
    lines = []
    for i, h in enumerate(hits, 1):
        r = records[h["rid"]]
        src = h.get("src", ""); src_tag = f" [{src}]" if src else ""
        score_show = h.get("score", h.get("score_raw", 0.0))
        lines.append(f"#{i:02d}  score={score_show:.2f}{src_tag}  [{r['channel']}/{r['file']}]")
        lines.append("     " + _preview_line(r))
    print(_box(title, "\n".join(lines) if lines else "(không có)"))

# ================= Cleans & helpers =================
_word = re.compile(r"\w+", flags=re.UNICODE)
def _tok(s: str) -> List[str]:
    if not s: return []
    return [w.lower() for w in _word.findall(s)]

def clean_text(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFC", s).lower()
    s = re.sub(r"[^a-z0-9\u00c0-\u024f\u1e00-\u1eff\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _extract_q_and_a(r: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    qa = r.get("qa") or {}
    q = qa.get("question") or r.get("question")
    a = qa.get("answer")   or r.get("answer")
    q = q.strip() if isinstance(q, str) and q.strip() else None
    a = a.strip() if isinstance(a, str) and a.strip() else None
    return q, a

def _extract_text(r: Dict[str, Any]) -> str:
    for key in ["text","content","search_text","body","chunk_text","paragraph"]:
        v = r.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    q, a = _extract_q_and_a(r)
    if q or a: return ("Q: " + (q or "") + (f"\nA: {a}" if a else ""))
    return ""

# ================= Signals (log) =================
_PAT = [
    r"\bđáp án\s*[:\-]", r"\btrả lời\s*[:\-]", r"\bkết luận\s*[:\-]",
    r"\bchọn đáp án\b", r"\bđúng là\b", r"\blà\s*[:\-]\s", r"\bđịnh nghĩa\s*[:\-]"
]
def _has_direct_answer(r: Dict[str, Any]) -> bool:
    qa = r.get("qa") or {}; ans = qa.get("answer") or r.get("answer")
    if isinstance(ans, str) and ans.strip(): return True
    txt = _extract_text(r)
    if not isinstance(txt, str) or not txt: return False
    for p in _PAT:
        if re.search(p, txt, flags=re.IGNORECASE): return True
    return False

def _evidence_score(records: List[Dict[str, Any]], hits: List[Dict[str, Any]]) -> float:
    if not hits: return 0.0
    return sum(1 for h in hits if _has_direct_answer(records[h["rid"]])) / max(1, len(hits))

# ================= Fusion / context helpers =================
def rrf_fusion(sem_hits, bm_hits, k: int = 60):
    rank_sem = {h["rid"]: i for i, h in enumerate(sorted(sem_hits, key=lambda x: -x["score"]), start=1)}
    rank_bm  = {h["rid"]: i for i, h in enumerate(sorted(bm_hits, key=lambda x: -x["score"]), start=1)}
    rids = set(list(rank_sem.keys()) + list(rank_bm.keys()))
    fused = []
    for rid in rids:
        r1 = rank_sem.get(rid, 10**9); r2 = rank_bm.get(rid, 10**9)
        fused.append({"rid": rid, "score": 1.0/(k+r1) + 1.0/(k+r2), "src": "rrf"})
    fused.sort(key=lambda x: -x["score"])
    return fused

def dedup_by_source(records, hits, max_per_source: int = 1):
    seen, kept = {}, []
    for h in hits:
        r = records[h["rid"]]; key = f"{r.get('channel','?')}::{r.get('file','?')}"
        if seen.get(key, 0) < max_per_source:
            kept.append(h); seen[key] = seen.get(key, 0) + 1
    return kept

_DEF_PAT = re.compile(r"\b(là gì|định nghĩa|ý nghĩa|ký hiệu|đáp án)\b", re.IGNORECASE)
def sort_by_definition_first(records, hits):
    def score(h): return 1 if _DEF_PAT.search((records[h["rid"]].get("text") or "").lower()) else 0
    return sorted(hits, key=score, reverse=True)

def join_adjacent_chunks(records, hits, window: int = 1, max_chars: int = 1200) -> List[str]:
    out, taken = [], set()
    for h in hits:
        rid = h["rid"]
        if rid in taken: continue
        r = records[rid]; file_id = r.get("file")
        pieces = [r.get("text") or ""]
        for off in range(1, window+1):
            for rrid in [rid-off, rid+off]:
                if 0 <= rrid < len(records) and rrid not in taken:
                    rr = records[rrid]
                    if rr.get("file") == file_id:
                        pieces.append(rr.get("text") or ""); taken.add(rrid)
        merged = " ".join([p.strip() for p in pieces if p and p.strip()])
        out.append(merged[:max_chars] + ("…" if len(merged) > max_chars else ""))
        taken.add(rid)
    return out

def _build_context_block_from_hits(ctx_hits: List[str], topn_ctx: int, max_chars_per_item: int, max_total_chars: int) -> str:
    blocks, total = [], 0
    for idx, text in enumerate(ctx_hits[:topn_ctx], 1):
        head = f"[{idx}]"
        body = (text or "").replace("\r"," ").replace("\t"," ").strip()
        if len(body) > max_chars_per_item: body = body[:max_chars_per_item] + "…"
        block = head + "\n" + body
        if total + len(block) > max_total_chars: break
        blocks.append(block); total += len(block)
    return "\n\n".join(blocks)

def _build_bm25_hints_block(records: List[Dict[str,Any]], bm25_hits: List[Dict[str,Any]], max_chars_per_item: int = 160) -> str:
    lines = []
    for i, h in enumerate(bm25_hits, 1):
        r = records[h["rid"]]
        q, a = _extract_q_and_a(r)
        txt = (q or a or _extract_text(r)).replace("\n", " ")
        if len(txt) > max_chars_per_item: txt = txt[:max_chars_per_item] + "…"
        lines.append(f"[B{i}] {txt}")
    return "\n".join(lines)

def _build_bm25_only_block(records: List[Dict[str, Any]], bm25_hits: List[Dict[str, Any]], topn: int = 3, max_chars: int = 700) -> str:
    blocks = []
    for i, h in enumerate(bm25_hits[:topn], 1):
        r = records[h["rid"]]
        q, a = _extract_q_and_a(r)
        txt = ("Q: " + (q or "") + ("\nA: " + a if a else "")).strip() if (q or a) else (_extract_text(r) or "").strip()
        if len(txt) > max_chars: txt = txt[:max_chars] + "…"
        blocks.append(f"[B{i}]\n{txt}")
    return "\n\n".join(blocks)

def _print_top3_bm25_QA(records: List[Dict[str, Any]], bm25_hits: List[Dict[str, Any]]):
    """Print top-3 BM25 hits as suggestion Q/A (fallback display)."""
    lines = []
    for i, h in enumerate(bm25_hits[:3], 1):
        r = records[h["rid"]]
        q, a = _extract_q_and_a(r)
        if q or a:
            prev_q = (q or "").replace("\n", " ")
            prev_a = (a or "").replace("\n", " ")
            if len(prev_q) > 160: prev_q = prev_q[:160] + "…"
            if len(prev_a) > 160: prev_a = prev_a[:160] + "…"
            lines.append(f"#{i:02d} [{r['channel']}/{r['file']}]")
            lines.append(f"  Q: {prev_q}")
            lines.append(f"  A: {prev_a}")
        else:
            txt = (_extract_text(r) or "").replace("\n", " ")
            if len(txt) > 180: txt = txt[:180] + "…"
            lines.append(f"#{i:02d} [{r['channel']}/{r['file']}]")
            lines.append(f"  {txt}")
    body = "\n".join(lines) if lines else "(không có)"
    print(_box("GỢI Ý CÂU HỎI", body))

# ================= Main =================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-root", type=str, default="embed_data")
    ap.add_argument("--channels", nargs="+", default=["dialogue","lesson"])

    # qa600.json (BM25-only)
    ap.add_argument("--qa600-path", type=str, default="data/qa600.json")
    ap.add_argument("--qa600-topn", type=int, default=3)
    ap.add_argument("--qa600-thr", type=float, default=0.9)

    # Semantic (embed)
    ap.add_argument("--model-id", type=str, default=os.getenv("EMB_MODEL","AITeamVN/Vietnamese_Embedding"))
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu"])

    # Retrieval (ALL = embed corpus)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--bm25-topn", type=int, default=3)
    ap.add_argument("--bm25-k1", type=float, default=1.5)
    ap.add_argument("--bm25-b", type=float, default=0.75)
    ap.add_argument("--sem-limit", type=int, default=3)

    # LLM
    ap.add_argument("--llm-off", action="store_true")

    # Mặc định dùng model Ollama đã tạo: phogpt4b-ft (PhoGPT base + LoRA adapter GGUF)
    ap.add_argument("--llm-model-id", type=str, default="phogpt4b-ft")
    ap.add_argument("--llm-temp", type=float, default=float(os.getenv("LLM_TEMP","0.3")))  # Tăng từ 0.2
    ap.add_argument("--llm-ollama-host", type=str, default=os.getenv("LLM_OLLAMA_HOST","http://localhost:11434"))
    # Tăng mặc định để sinh câu trả lời dài hơn (có thể override bằng env hoặc CLI)
    ap.add_argument("--llm-max-new", type=int, default=int(os.getenv("LLM_MAX_NEW","768")))

    # LLM Provider (Ollama-only)
    ap.add_argument("--llm-provider", type=str, default="ollama",
                    choices=["ollama"])

    ap.add_argument("--session-id", type=str, default="cli",
                help="ID phiên chat để lưu/đọc memory (file JSONL)")
    ap.add_argument("--memory-max-turns", type=int, default=4,
                    help="Số lượt hội thoại gần nhất đưa vào prompt")
    ap.add_argument("--memory-max-chars", type=int, default=1200,
                    help="Giới hạn ký tự history nhúng vào prompt")
    

    # Simple Prompting Options
    ap.add_argument("--enable-simple-prompting", action="store_true", default=True,
                    help="Bật simple prompting system (default: True)")
    ap.add_argument("--prompt-temperature", type=float, default=0.3,
                    help="Temperature cho prompting system")
    ap.add_argument("--enable-answer-preface", action="store_true", default=True,
                    help="Thêm câu mở đầu lịch sự trước câu trả lời")

    

    # Context
    ap.add_argument("--ctx-topn", type=int, default=3)
    ap.add_argument("--ctx-max-chars", type=int, default=16000)
    ap.add_argument("--ctx-join-window", type=int, default=0)
    ap.add_argument("--ctx-item-max-chars", type=int, default=2000,
                    help="Giới hạn ký tự mỗi đoạn context")

    # Thresholds
    ap.add_argument("--thr-warn", type=float, default=0.5)

    # Query gate
    ap.add_argument("--min-query-chars", type=int, default=4)
    ap.add_argument("--min-keyword-len", type=int, default=2)

    # Debug
    ap.add_argument("--no-print-prompt", action="store_true")
    ap.add_argument("--max-preview-len", type=int, default=180)

    # One shot
    ap.add_argument("--once", type=str, default=None)

    args = ap.parse_args()

    print(f"[INFO] LLM Provider: {args.llm_provider}")
    print(f"[INFO] LLM Model ID: {args.llm_model_id}")
    print(f"[INFO] Simple Prompting: {'✅ Enabled' if args.enable_simple_prompting else '❌ Disabled'}")

    # 1) Load embedded corpus (semantic)
    index, records = load_corpus(root=args.embed_root, channels=tuple(args.channels))
    sem = SemanticSearcher(index, model_id=args.model_id, device=args.device)

    # 2) BM25 over ALL embedded records
    bm25_docs_all = []
    for r in records:
        q, a = _extract_q_and_a(r)
        bm25_docs_all.append("\n".join([x for x in [q, a] if x]) if (q or a) else _extract_text(r))
    bm25_all = BM25(bm25_docs_all, k1=args.bm25_k1, b=args.bm25_b)

    # 3) Load qa600.json (BM25-only)
    qa_records, qa_docs = [], []
    if args.qa600_path and os.path.isfile(args.qa600_path):
        try:
            with open(args.qa600_path, "r", encoding="utf-8") as f:
                items = json.load(f)
            if isinstance(items, dict): items = [items]
            for it in items:
                q = (it.get("question") or "").strip()
                a = (it.get("answer")   or it.get("answer_text") or "").strip()
                if not (q or a): continue
                qa_records.append({
                    "text": (q + ("\n" + a if a else "")).strip(),
                    "qa": {"question": q or None, "answer": a or None},
                    "meta": {"order": it.get("order"), "importance": it.get("importance", False)},
                    "file": os.path.basename(args.qa600_path),
                    "channel": "qa600",
                })
                qa_docs.append("\n".join([x for x in [q, a] if x]))
            print(f"[OK] Loaded {len(qa_records)} QA from {args.qa600_path} (BM25-first).")
        except Exception as e:
            print(f"[WARN] Cannot read qa600 at {args.qa600_path}: {e}")
    else:
        print(f"[WARN] qa600 not found at: {args.qa600_path}")
    bm25_qa = BM25(qa_docs, k1=args.bm25_k1, b=args.bm25_b) if qa_docs else None

    # 4) LLM
    llm = None

    if not args.llm_off:
        print(f"[INFO] Initializing LLM with provider: {args.llm_provider}")
        llm = build_llm_from_args(args)

    def run_query(q: str):
        # Preprocess user input (normalize spacing, units, remove special chars)
        orig_q = (q or "").strip()
        q = clean_user_input(orig_q)
        if len(q) < args.min_query_chars:
            print(_box("CẢNH BÁO", f"Câu hỏi quá ngắn (< {args.min_query_chars} ký tự)."))
            return

        # Resolve follow-up using last focus (if any) BEFORE searching
        try:
            q_resolved = resolve_after_preprocess(q, args.session_id)
        except Exception:
            q_resolved = q
        q_proc = clean_text(q_resolved)
        if not q_proc or len(q_proc) < 2:
            print(_box("CẢNH BÁO", "Câu hỏi không hợp lệ sau khi xử lý."))
            return

        # Show both original and post-processed input for transparency
        display = f"Nguyên gốc: {orig_q}\n\nSau tiền xử lý: {q}"
        print(_box("CÂU HỎI", display))

        # Small-talk / identity handling
        if _is_smalltalk_identity(orig_q) or _is_smalltalk_identity(q):
            print(_box("TRẢ LỜI", _HP_AI_REPLY)); print(); return

        # ---------- 1) BM25 trên QA600 trước ----------
        if bm25_qa is not None:
            qa_hits = bm25_qa.search(q_proc, topk=max(1, args.qa600_topn), min_keyword_len=args.min_keyword_len)
            _print_hits("KẾT QUẢ BM25 (qa600)", qa_records, qa_hits)
            if qa_hits and qa_hits[0]["score"] >= args.qa600_thr:
                top = qa_hits[0]; rec = qa_records[top["rid"]]
                ans = (rec.get("qa") or {}).get("answer") or _extract_text(rec) or ""

                print(_box("TRẢ LỜI (CHẮC CHẮN)", _clean_answer(ans))); print(); return

        # ---------- 2) Semantic + BM25(all) ----------
        se_hits = sem.search(q_proc, topk=args.topk)
        # Prioritize semantic hits coming from `lesson` channel.
        try:
            lesson_hits = [h for h in se_hits if (records[h["rid"]].get("channel") == "lesson")]
            other_hits = [h for h in se_hits if (records[h["rid"]].get("channel") != "lesson")]
            # Preserve original ordering within each group (already by score desc from search)
            se_hits = lesson_hits + other_hits
        except Exception:
            # If anything goes wrong, keep original se_hits
            pass
        bm_hits = bm25_all.search(q_proc, topk=args.bm25_topn, min_keyword_len=args.min_keyword_len)
        mx = max([h["score"] for h in se_hits], default=0.0)

        # Update memory focus if semantic confidence is high enough
        try:
            update_focus_after_sem(q_resolved, args.session_id, sem_max=mx, threshold=args.thr_warn)
        except Exception:
            pass

        ev = _evidence_score(records, se_hits)

        print(_box("TÍN HIỆU TIN CẬY", f"SEM_max={mx:.2f} | Evidence≈{ev:.2f}"))
        _print_hits("KẾT QUẢ SEMANTIC", records, se_hits)
        _print_hits("KẾT QUẢ BM25 ", records, bm_hits)

        if args.llm_off:
            print(_box("LLM", "LLM đang tắt (--llm-off).")); print(); return

        # --- semantic >= 0.5 => chắc chắn ---
        if mx >= 0.5 and llm:
            # Chỉ dùng SEMANTIC hits, không RRF fusion
            se_sorted = sorted(se_hits, key=lambda x: -x["score"])
            se_dedup = dedup_by_source(records, se_sorted, max_per_source=2)
            se_def_first = sort_by_definition_first(records, se_dedup)
            ctx_ids = se_def_first[:args.sem_limit]
            
            # Tạo context chính từ semantic hits
            if args.ctx_join_window > 0:
                semantic_blocks = join_adjacent_chunks(records, ctx_ids, window=args.ctx_join_window, max_chars=args.ctx_item_max_chars)
            else:
                semantic_blocks = [(records[h["rid"]].get("text") or "")[:args.ctx_item_max_chars] for h in ctx_ids]

            main_context = _build_context_block_from_hits(semantic_blocks, args.ctx_topn, args.ctx_item_max_chars, args.ctx_max_chars)

            # BM25 chỉ làm context bổ sung (hints)
            bm25_context = ""
            if bm_hits:
                hints = _build_bm25_hints_block(records, bm_hits[:1], max_chars_per_item=400)
                if hints.strip():
                    bm25_context = f"\n\n[THAM KHẢO BM25]\n{hints}"

            # Display context combines semantic + BM25 hints, but LLM should only see semantic
            display_context = main_context + bm25_context
            final_context_for_llm = main_context

            if not args.no_print_prompt:
                print(_box("CONTEXT (SEMANTIC + BM25 HINTS)", display_context or "(rỗng)"))

            try:
                if args.enable_simple_prompting:
                    answer = simple_generate_answer(
                        llm, q_resolved, final_context_for_llm, 
                        max_new_tokens=args.llm_max_new, 
                        temperature=args.prompt_temperature
                    )
                else:
                    answer = generate_answer(llm, q_resolved, final_context_for_llm, max_new_tokens=args.llm_max_new, temperature=args.llm_temp)
            except Exception as e:
                print(_box("TRẢ LỜI", f"[LLM ERROR] {e}"));
                import traceback; traceback.print_exc()
                return
            # Enrich short answers and optionally add a polite preface
            answer_final = _enrich_short_answer((answer or "").strip(), display_context)
            answer_final = _maybe_with_preface(answer_final, orig_q, args.enable_answer_preface)
            print(_box("TRẢ LỜI (CHẮC CHẮN)", (_clean_answer(answer_final) or "Thông tin chưa đủ."))); print(); return

        # --- semantic < 0.5 ---
        if not bm_hits:
            print(_box("KẾT LUẬN", "Không tìm thấy căn cứ đủ mạnh và không có gợi ý từ BM25.")); print(); return

        # Prefer BM25 suggestions from the `dialogue` channel (user request).
        # If no dialogue hits are available, fall back to the full BM25 results.
        try:
            bm_dialogue_hits = [h for h in bm_hits if records[h["rid"]].get("channel") == "dialogue"]
        except Exception:
            bm_dialogue_hits = []

        chosen_bm_hits = bm_dialogue_hits if bm_dialogue_hits else bm_hits

        bm25_block = _build_bm25_only_block(records, chosen_bm_hits, topn=3, max_chars=700)
        print(_box(f"CÂU HỎI GỢI Ý", bm25_block or "(rỗng)"))


    # Run once / REPL
    if args.once:
        run_query(args.once); return

    print(_box("HƯỚNG DẪN", "Nhập câu hỏi về lý thuyết lái xe (Enter để thoát). Ví dụ: 'Biển P.130 là gì?'"))
    while True:
        try:
            q = input("Q> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not q: break
        run_query(q)

if __name__ == "__main__":
    main()
