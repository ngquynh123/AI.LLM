
# # -*- coding: utf-8 -*-
# from typing import Dict
# from preprocess import clean_user_input
# import re
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_core.chat_history import BaseChatMessageHistory
# from langchain_community.chat_models import ChatOllama

# try:
#     from langchain_core.chat_history import InMemoryChatMessageHistory  # >=0.2
# except Exception:
#     from langchain_community.chat_message_histories import ChatMessageHistory as InMemoryChatMessageHistory

# _SESSION_STORE: Dict[str, InMemoryChatMessageHistory] = {}

# def get_session_history(session_id: str) -> BaseChatMessageHistory:
#     if session_id not in _SESSION_STORE:
#         _SESSION_STORE[session_id] = InMemoryChatMessageHistory()
#     return _SESSION_STORE[session_id]

# def build_memory_chain(model_id: str, base_url: str = "http://localhost:11434", temperature: float = 0.2):
#     llm = ChatOllama(model=model_id, base_url=base_url, temperature=temperature)
#     prompt = ChatPromptTemplate.from_messages([
#         ("system",
#          "Bạn là trợ lý AI về LÝ THUYẾT LÁI XE Ô TÔ tại Việt Nam. "
#          "Chỉ dựa vào NỘI DUNG THAM CHIẾU nếu có. Trả lời ngắn gọn, trung lập."),
#         MessagesPlaceholder(variable_name="history"),
#         ("system", "NỘI DUNG THAM CHIẾU (nếu có):\n{context}"),
#         ("human", "{question}"),
#     ])
#     chain = prompt | llm
#     with_history = RunnableWithMessageHistory(
#         chain,
#         get_session_history,
#         input_messages_key="question",
#         history_messages_key="history"
#     )
#     return with_history


# # ------------------
# # Simple HF memory chain
# # ------------------
# _HF_SESSION_STORE: Dict[str, list] = {}

# def get_hf_session_history(session_id: str) -> list:
#     """Return a list of turns for HF memory chain. Each turn is dict {user, assistant}."""
#     if session_id not in _HF_SESSION_STORE:
#         _HF_SESSION_STORE[session_id] = []
#     return _HF_SESSION_STORE[session_id]


# def build_memory_chain_hf(llm, max_turns: int = 2, max_chars: int = 1200):
#     """Create a lightweight memory chain wrapper for local HF-style LLMs.

#     The returned object exposes an `invoke(inputs, config=None)` method and
#     returns an object with a `content` attribute (or stringifiable) so it's
#     compatible with how `main.py` reads memory chain output.
#     """
#     class HFMemoryChain:
#         def __init__(self, llm, max_turns=4, max_chars=1200):
#             self.llm = llm
#             self.max_turns = int(max_turns)
#             self.max_chars = int(max_chars)

#         def invoke(self, inputs: dict, config: dict = None):
#             # inputs: {"question": str, "context": str}
#             cfg = config or {}
#             sess = cfg.get("configurable", {})
#             session_id = sess.get("session_id") if isinstance(sess, dict) else None
#             session_id = session_id or "cli"

#             question = (inputs.get("question") or inputs.get("input") or "").strip()
#             context = (inputs.get("context") or "").strip()

#             # Preprocess user input with same preprocessing used elsewhere
#             try:
#                 question_proc = clean_user_input(question)
#             except Exception:
#                 question_proc = question.strip()

#             # Expand follow-up / ellipsis questions using recent history
#             question_expanded = self._expand_followup(question_proc, session_id)
#             if question_expanded:
#                 question = question_expanded
#             else:
#                 question = question_proc

#             history = get_hf_session_history(session_id)

#             # Build history text (take last max_turns)
#             pieces = []
#             for turn in history[-self.max_turns:]:
#                 u = turn.get("user", "")
#                 a = turn.get("assistant", "")
#                 if u:
#                     pieces.append(f"Người dùng: {u}")
#                 if a:
#                     pieces.append(f"Trợ lý: {a}")
#             history_text = "\n".join(pieces)

#             # Compose prompt: history + context + question
#             prompt_parts = []
#             if history_text:
#                 prompt_parts.append("Lịch sử hội thoại:\n" + history_text)
#             if context:
#                 # Truncate context to max_chars to avoid overly long prompts
#                 ctx = context if len(context) <= self.max_chars else context[-self.max_chars:]
#                 prompt_parts.append("NỘI DUNG THAM KHẢO:\n" + ctx)
#             prompt_parts.append("CÂU HỎI:\n" + question)
#             prompt = "\n\n".join(prompt_parts)

#             # Call the LLM
#             try:
#                 if hasattr(self.llm, "generate"):
#                     # Some local LLM wrappers accept (prompt, stop=...)
#                     answer = self.llm.generate(prompt)
#                 else:
#                     # Fallback: try calling as a callable
#                     answer = str(self.llm)
#             except Exception as e:
#                 answer = f"[MEMORY_LLM_ERROR] {e}"

#             # Save turn into history
#             try:
#                 history.append({"user": question, "assistant": answer})
#                 # Trim to last max_turns
#                 if len(history) > self.max_turns:
#                     del history[0: len(history) - self.max_turns]
#             except Exception:
#                 pass

#             # Return an object with .content attribute for compatibility
#             class Out:
#                 def __init__(self, content):
#                     self.content = content
#                 def __str__(self):
#                     return self.content

#             return Out(answer)

#     return HFMemoryChain(llm, max_turns=max_turns, max_chars=max_chars)


# -*- coding: utf-8 -*-
from typing import Dict, Optional, List, Tuple
import re

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_models import ChatOllama

try:
    from langchain_core.chat_history import InMemoryChatMessageHistory  # >=0.2
except Exception:
    from langchain_community.chat_message_histories import ChatMessageHistory as InMemoryChatMessageHistory

# ----------------------------
# Stores
# ----------------------------
_SESSION_STORE: Dict[str, InMemoryChatMessageHistory] = {}
_STATE_STORE: Dict[str, Dict] = {}  # {session_id: {"last_focus": str, "turns": list}}

# ----------------------------
# Helpers for focus extraction / resolution
# ----------------------------
_VN_STOP = {
    "là","gì","của","về","theo","trong","định","nghĩa","khái","niệm",
    "các","những","này","kia","đó","nó","?",":",",",".","thế","nào",
    "sắp","xếp","theo","thứ","tự","có","mấy","loại","bao","nhiêu"
}

def _normalize(s: str) -> str:
    return " ".join((s or "").strip().split())

def _toks(s: str) -> List[str]:
    return re.findall(r"[0-9A-Za-zÀ-ỹ_]+", s or "", flags=re.UNICODE)

def _extract_focus_from_question(q: str) -> str:
    # Lấy cụm danh từ “trọng tâm” làm focus (thô, nhanh)
    toks = _toks(q)
    keep = [t for t in toks if t.lower() not in _VN_STOP and len(t) >= 2]
    return " ".join(keep[:6]).strip() or _normalize(q)

def _state(session_id: str) -> Dict:
    st = _STATE_STORE.get(session_id)
    if not st:
        st = {"last_focus": None, "turns": []}
        _STATE_STORE[session_id] = st
    return st

def _has_tail(q: str, kw: str) -> bool:
    ql = (q or "").lower()
    m = re.search(rf"{kw}\s+(của\s+)?([a-zA-ZÀ-ỹ0-9_ ]{{3,}})$", ql)
    if not m: return False
    tail = (m.group(2) or "").strip()
    return tail not in {"nó","đó","này","kia"}

def _resolve_with_focus(q: str, last_focus: Optional[str]) -> Tuple[str, bool]:
    qn = _normalize(q)
    if not last_focus:
        return qn, False
    ql = qn.lower()

    # Nếu user đã ghi rõ đối tượng sau từ khoá, không sửa nữa
    if _has_tail(qn, "khái niệm") or _has_tail(qn, "định nghĩa") or _has_tail(qn, "chức năng"):
        return qn, False

    changed = False

    # Khái niệm / Định nghĩa
    if "khái niệm của nó" in ql:
        qn = f"khái niệm {last_focus}"; changed = True
    elif "khái niệm" in ql and not _has_tail(qn, "khái niệm"):
        qn = f"khái niệm {last_focus}"; changed = True
    elif "định nghĩa của nó" in ql:
        qn = f"định nghĩa {last_focus}"; changed = True
    elif "định nghĩa" in ql and not _has_tail(qn, "định nghĩa"):
        qn = f"định nghĩa {last_focus}"; changed = True

    # Chức năng
    elif "chức năng" in ql and not _has_tail(qn, "chức năng"):
        qn = f"chức năng {last_focus}"; changed = True

    # Sắp theo thứ tự
    elif "sắp xếp theo thứ tự" in ql or "sắp theo thứ tự" in ql:
        qn = f"sắp xếp {last_focus} theo thứ tự"; changed = True

    # Có mấy bánh
    elif "có mấy bánh" in ql:
        qn = f"{last_focus} có mấy bánh"; changed = True

    # ‘nó là gì / đó là gì / của nó là gì’
    elif "nó là gì" in ql or "đó là gì" in ql or "của nó là gì" in ql:
        qn = f"{last_focus} là gì"; changed = True

    return _normalize(qn), changed

# ----------------------------
# PUBLIC API cho main.py
# ----------------------------
def resolve_after_preprocess(question: str, session_id: str) -> str:
    """
    Gọi SAU bước clean/tiền xử lý. Tự bù ngữ cảnh dựa trên last_focus đã nhớ.
    """
    st = _state(session_id)
    resolved, _ = _resolve_with_focus(_normalize(question), st.get("last_focus"))
    return resolved

def update_focus_after_sem(question_used: str, session_id: str, sem_max: float = None, threshold: float = 0.55):
    """
    Gọi SAU khi đã tính SEM_max cho câu hỏi ĐÃ BÙ.
    - Nếu sem_max >= threshold -> lưu focus (chắc chắn).
    - Nếu sem_max is None -> cập nhật mềm (tuỳ bạn dùng).
    """
    st = _state(session_id)
    try:
        if isinstance(sem_max, (int, float)) and float(sem_max) >= float(threshold):
            st["last_focus"] = _extract_focus_from_question(question_used)
        elif sem_max is None and question_used:
            st["last_focus"] = _extract_focus_from_question(question_used)
    except Exception:
        pass

# ----------------------------
# LangChain chain (giữ API cũ, tự resolve trước khi gọi model)
# ----------------------------
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _SESSION_STORE:
        _SESSION_STORE[session_id] = InMemoryChatMessageHistory()
    return _SESSION_STORE[session_id]

def build_memory_chain(model_id: str, base_url: str = "http://localhost:11434", temperature: float = 0.2):
    llm = ChatOllama(model=model_id, base_url=base_url, temperature=temperature)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Bạn là trợ lý AI về LÝ THUYẾT LÁI XE Ô TÔ tại Việt Nam. "
         "Chỉ dựa vào NỘI DUNG THAM CHIẾU nếu có. Trả lời ngắn gọn, trung lập."),
        MessagesPlaceholder(variable_name="history"),
        ("system", "NỘI DUNG THAM CHIẾU (nếu có):\n{context}"),
        ("human", "{question}"),
    ])
    chain = prompt | llm

    with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history"
    )

    class _ResolvedWrapper:
        def __init__(self, runnable):
            self.runnable = runnable

        def invoke(self, inputs: dict, config: dict = None):
            cfg = config or {}
            sess = (cfg.get("configurable") or {}) if isinstance(cfg, dict) else {}
            session_id = (sess.get("session_id") if isinstance(sess, dict) else None) or "cli"

            raw_q = _normalize(inputs.get("question") or inputs.get("input") or "")
            st = _state(session_id)
            q_resolved, _ = _resolve_with_focus(raw_q, st.get("last_focus"))
            payload = dict(inputs); payload["question"] = q_resolved

            return self.runnable.invoke(payload, config=cfg)

    return _ResolvedWrapper(with_history)

# ----------------------------
# HF-style memory chain (local HF)
# ----------------------------
def build_memory_chain_hf(llm, max_turns: int = 4, max_chars: int = 1200):
    class HFMemoryChain:
        def __init__(self, llm, max_turns=4, max_chars=1200):
            self.llm = llm
            self.max_turns = int(max_turns)
            self.max_chars = int(max_chars)

        def invoke(self, inputs: dict, config: dict = None):
            cfg = config or {}
            sess = (cfg.get("configurable") or {}) if isinstance(cfg, dict) else {}
            session_id = (sess.get("session_id") if isinstance(sess, dict) else None) or "cli"

            st = _state(session_id)
            turns = st["turns"]

            raw_q = _normalize(inputs.get("question") or inputs.get("input") or "")
            ctx = (inputs.get("context") or "").strip()

            # 1) resolve theo last_focus để prompt có ngữ cảnh
            q_resolved, _ = _resolve_with_focus(raw_q, st.get("last_focus"))

            # 2) build history
            lines = []
            for t in turns[-self.max_turns:]:
                u, a = t.get("user", ""), t.get("assistant", "")
                if u: lines.append(f"Người dùng: {u}")
                if a: lines.append(f"Trợ lý: {a}")
            history_text = "\n".join(lines)

            parts = []
            if history_text:
                parts.append("Lịch sử hội thoại:\n" + history_text)
            if ctx:
                ctx_clip = ctx if len(ctx) <= self.max_chars else ctx[-self.max_chars:]
                parts.append("NỘI DUNG THAM KHẢO:\n" + ctx_clip)
            parts.append("CÂU HỎI:\n" + q_resolved)
            prompt = "\n\n".join(parts)

            try:
                if hasattr(self.llm, "generate"):
                    answer = self.llm.generate(prompt)
                else:
                    answer = str(self.llm)
            except Exception as e:
                answer = f"[MEMORY_LLM_ERROR] {e}"

            try:
                turns.append({"user": q_resolved, "assistant": answer})
                if len(turns) > self.max_turns:
                    del turns[0: len(turns) - self.max_turns]
            except Exception:
                pass

            class Out:
                def __init__(self, content): self.content = content
                def __str__(self): return self.content
            return Out(answer)

    return HFMemoryChain(llm, max_turns=max_turns, max_chars=max_chars)
