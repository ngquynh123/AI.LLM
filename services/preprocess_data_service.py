"""Dịch vụ tiền xử lý dữ liệu văn bản (interactive).

Chương trình này cung cấp:
- `preprocess_text(text)`: chuẩn hoá chuỗi (unicode, chuyển chữ thường, loại ký tự không cần thiết)

CLI hiện ở chế độ tương tác: khi chạy `python service/preprocess_data.py`
người dùng sẽ được hỏi nhập câu hỏi hoặc dòng văn bản, chương trình in kết quả đã
tiền xử lý ngay lập tức. Nhấn Enter trên dòng rỗng để thoát.

Ví dụ:
    python service/preprocess_data.py
    > Hôm nay @ là thứ mấy ???
    hôm nay là thứ mấy ?
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Optional


def preprocess_text(text: Optional[str]) -> str:
    """Tiền xử lý một chuỗi văn bản.

    Hành vi:
    - Chuẩn hoá unicode, đơn vị, và loại ký tự không mong muốn
    - Chuyển chữ hoa về chữ thường
    - Thu gọn dấu câu lặp và ký tự lặp
    """
    if text is None:
        return ""
    s = clean_user_input(text)
    s = s.lower()
    # Thu gọn dấu câu lặp (???, !!!, ...)
    s = re.sub(r"([?!\.])\1+", r"\1", s)
    # Thu gọn ký tự lặp liên tiếp (ví dụ: hoommm -> hom)
    s = re.sub(r"([\w])\1+", r"\1", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


# --- Begin helpers copied/adapted from retriever/preprocess.py ---
_ALLOWED_PUNCT = ".,:;?()[]{}-/%°+\"'"

_UNIT_NORMALIZATIONS = [
    (r"km\s*[/\\\-\s]*h|kmh|km\s*h|km/s|km\s*per\s*s", "km/h"),
    (r"m\s*[/\\\-\s]*s|m/s|ms", "m/s"),
    (r"km\s*[/\\\-\s]*s|kms", "km/s"),
    (r"%", "%"),
]


def _normalize_units(s: str) -> str:
    out = s
    for pat, rep in _UNIT_NORMALIZATIONS:
        out = re.sub(pat, rep, out, flags=re.IGNORECASE)
    return out


def clean_user_input(text: Optional[str]) -> str:
    """Chuẩn hoá unicode, đơn vị và loại bỏ các ký tự không mong muốn.

    Trả về chuỗi đã được làm sạch, phù hợp để xử lý tiếp theo.
    """
    if text is None:
        return ""
    s = unicodedata.normalize("NFC", str(text)).strip()
    # thu gọn khoảng trắng và newline
    s = re.sub(r"[\t\n\r]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    # chuẩn hoá đơn vị trước
    s = _normalize_units(s)
    # cho phép chữ, số và dấu câu đã chọn; thay các ký tự khác bằng khoảng trắng
    allowed = re.escape(_ALLOWED_PUNCT)
    s = re.sub(rf"[^\w\s{allowed}]", " ", s, flags=re.UNICODE)
    # loại bỏ khoảng trắng quanh dấu gạch chéo và chuẩn hoá khoảng cách số-đơn vị
    s = re.sub(r"\s*/\s*", "/", s)
    s = re.sub(r"(\d)(km/h|m/s|km/s|%)\b", r"\1 \2", s, flags=re.IGNORECASE)
    return s.strip().lower()

# --- End helpers ---


def preprocess_json(obj: Any) -> Any:
    """Đệ quy tiền xử lý mọi giá trị chuỗi trong cấu trúc tương tự JSON.

    Các giá trị không phải chuỗi (số, bool, None) giữ nguyên.
    """
    if isinstance(obj, str):
        return preprocess_text(obj)
    if isinstance(obj, list):
        return [preprocess_json(x) for x in obj]
    if isinstance(obj, dict):
        return {k: preprocess_json(v) for k, v in obj.items()}
    # các kiểu khác (int/float/bool/None) giữ nguyên
    return obj

def normalize_text(text: Optional[str], remove_diacritics: bool = False) -> str:
    """Chuẩn hoá chuỗi thêm:

    - trim, collapse whitespace
    - chuyển chữ thường
    - loại bỏ dấu (tùy chọn) để so sánh không dấu
    - thu gọn dấu câu/ký tự lặp
    """
    if text is None:
        return ""
    s = str(text).strip()
    if remove_diacritics:
        # chuyển về NFD rồi bỏ dấu kết hợp
        s = unicodedata.normalize("NFD", s)
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    # loại bỏ ký tự lạ, chỉ giữ chữ số/chữ cái và các dấu câu phổ biến
    s = re.sub(r"[^\w\s\.,:;?()\[\]{}\-/%°+\'\"]", " ", s, flags=re.UNICODE)
    # thu gọn lặp ký tự và khoảng trắng
    s = re.sub(r"([?!\.])\1+", r"\1", s)
    s = re.sub(r"([\w])\1+", r"\1", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Module chỉ cung cấp các hàm tiền xử lý.
__all__ = [
    "preprocess_text",
    "clean_user_input",
    "preprocess_json",
    "normalize_text",
]
