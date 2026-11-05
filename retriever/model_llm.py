# model_llm.py


from typing import List, Optional
import os, re

# =========================
# Base interface
# =========================
class BaseLLM:
    def generate(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        raise NotImplementedError

# (Removed HF local model support; keep Ollama-only path)

# =========================
# Ollama LLM (unchanged)
# =========================
import requests

class OllamaGenerateLLM(BaseLLM):
    def __init__(self, model_id: str = None, host: str = None, temperature: float = 0.2, 
                 top_p: float = 0.95, max_new_tokens: int = 384, timeout_connect: int = 10,
                 timeout_read: int = 240, keep_alive: str = "5m"):
        # Mặc định dùng model Ollama đã tạo: phogpt4b-ft
        self.model_id = model_id or os.getenv("LLM_MODEL_ID", "phogpt4b-ft")
        self.host = (host or os.getenv("LLM_OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_new_tokens = int(max_new_tokens)
        self.timeout_connect = int(timeout_connect)
        self.timeout_read = int(timeout_read)
        self.keep_alive = keep_alive
        # Debug capture
        self.last_prompt = None
        self.last_output = None

    def set_params(self, temperature: float = None, max_new_tokens: int = None):
        if temperature is not None:
            self.temperature = float(temperature)
        if max_new_tokens is not None:
            self.max_new_tokens = int(max_new_tokens)

    def generate(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # record prompt for debugging
        try:
            self.last_prompt = prompt
        except Exception:
            self.last_prompt = None
        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_predict": self.max_new_tokens,
            },
        }
        if stop:
            payload["stop"] = stop
        try:
            r = requests.post(url, json=payload,
                              timeout=(self.timeout_connect, self.timeout_read))
            r.raise_for_status()
            data = r.json()
            # Capture the full raw JSON response for debugging. Also extract the primary text
            # response if present under the common 'response' key for backwards compatibility.
            try:
                self.last_output = data
            except Exception:
                # Fallback: store the textual body
                try:
                    self.last_output = r.text
                except Exception:
                    self.last_output = None
            out = ""
            try:
                out = (data.get("response") or "")
            except Exception:
                out = ""
            return (out or "").strip()
        except requests.exceptions.RequestException as e:
            return f"[Ollama ERROR] {e}"
        except Exception as e:
            return f"[Ollama ERROR] {e}"

# =========================
# Builders with error handling
# =========================
def build_llm_from_args(args) -> BaseLLM:
    """Always build an Ollama LLM instance (HF local removed)."""
    return OllamaGenerateLLM(
        model_id=args.llm_model_id,
        host=args.llm_ollama_host,
        temperature=args.llm_temp,
        top_p=0.95,
        max_new_tokens=args.llm_max_new,
        timeout_connect=getattr(args, "llm_timeout_connect", 10),
        timeout_read=getattr(args, "llm_timeout_read", 600),
        keep_alive=getattr(args, "llm_keep_alive", "5m"),
    )

def build_llm(*args, **kwargs):
    """Compatibility builder that returns an Ollama LLM (HF/local removed)."""
    return OllamaGenerateLLM(
        model_id=kwargs.get("model_id"),
        host=kwargs.get("host") or kwargs.get("base_url"),
        temperature=kwargs.get("temperature", 0.2),
        max_new_tokens=kwargs.get("max_new_tokens", 384),
    )

# =========================
# STOP tokens & utilities
# =========================
STOP_TOKENS = [
    "<sources>", "</sources>", "<bm25_hints>", "</bm25_hints>",
    "<bm25_only>", "</bm25_only>", "<question>", "</question>",
    "<sample_answer>", "</sample_answer>", "<theory>", "</theory>",
    "<given_answer>", "</given_answer>", "\n[B", "GỢI Ý TỪ BM25", "PROMPT ("
]

_TAGS_RE = re.compile(r"</?(sources|bm25_hints|bm25_only|question|sample_answer|theory|given_answer)>", re.IGNORECASE)
_BM25_LINE_RE = re.compile(r"^\s*\[B\d+\].*$", flags=re.MULTILINE)

def sanitize(text: str) -> str:
    s = text if isinstance(text, str) else (str(text) if text is not None else "")
    s = _BM25_LINE_RE.sub("", s)
    s = _TAGS_RE.sub("", s)
    return s.strip()

# =========================
# EMERGENCY prompt helpers
# =========================
def build_prompt_ref_with_sample(user_question: str, theory_block: str, sample_answer: Optional[str] = None) -> str:
    """EMERGENCY: Ultra-simple prompt"""
    question = user_question[:15] if user_question else "Hỏi"
    return f"Hỏi: {question}"

def build_prompt_with_context(user_question: str, theory_block: str) -> str:
    return f"Hỏi: {user_question[:15]}"

def build_prompt_with_sample(user_question: str, sample_answer: str) -> str:
    return f"Hỏi: {user_question[:15]}"

def build_prompt_direct(user_question: str) -> str:
    return f"Hỏi: {user_question[:15]}"

def generate_answer(llm, question: str, context: str, max_new_tokens: int = 150, temperature: Optional[float] = None) -> str:
    """Enhanced generate_answer function"""
    try:
        if not hasattr(llm, 'generate'):
            return "LLM không hoạt động"
        
        # Set parameters if provided
        if hasattr(llm, 'set_params'):
            params = {}
            if temperature is not None:
                params['temperature'] = temperature
            if max_new_tokens is not None:
                params['max_new_tokens'] = max_new_tokens
            
            if params:
                llm.set_params(**params)
        
        # Build a simple but effective prompt
        if context and context.strip():
            prompt = f"Câu hỏi: {question}\n\nThông tin tham khảo:\n{context[:1000]}\n\nTrả lời:"
        else:
            prompt = f"Câu hỏi: {question}\n\nTrả lời:"
        
        answer = llm.generate(prompt, stop=STOP_TOKENS)
        return answer.strip() if answer else "Không có câu trả lời"
        
    except Exception as e:
        print(f"[ERROR] generate_answer failed: {e}")
        return f"Lỗi sinh câu trả lời: {str(e)}"
