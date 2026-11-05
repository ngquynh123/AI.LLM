"""User input preprocessing helpers.

Provides `clean_user_input(text: str) -> str` that:
- strips leading/trailing whitespace
- normalizes unicode
- removes (or replaces) special characters except punctuation useful for meaning
- normalizes multiple spaces into single space
- fixes common unit spacing like 'km / h', 'km/s', 'kmh', 'km s' -> 'km/h'
- normalizes number formatting like '  50  km / h' -> '50 km/h'

You can extend this file with domain-specific normalizations.
"""
import re, unicodedata
from typing import Optional

# Patterns for units we want to normalize (map possible variants -> canonical)
_UNIT_NORMALIZATIONS = [
    (r"km\s*[/\\\-\s]*h|kmh|km\s*h|km/s|km\s*per\s*s", "km/h"),
    (r"m\s*[/\\\-\s]*s|m/s|ms", "m/s"),
    (r"km\s*[/\\\-\s]*s|kms", "km/s"),
    (r"%", "%"),
]

# Characters that are allowed in the cleaned input (letters, numbers, basic punctuation)
_ALLOWED_PUNCT = ".,:;?()[]{}-/%°+"


def _normalize_units(s: str) -> str:
    out = s
    for pat, rep in _UNIT_NORMALIZATIONS:
        out = re.sub(pat, rep, out, flags=re.IGNORECASE)
    return out


def clean_user_input(text: Optional[str]) -> str:
    if text is None:
        return ""
    # Normalize unicode and strip
    s = unicodedata.normalize("NFC", text)
    s = s.strip()

    # Replace repeated whitespace/newlines with single space
    s = re.sub(r"[\t\n\r]+", " ", s)
    s = re.sub(r"\s+", " ", s)

    # Normalize units first (e.g., 'km / h' -> 'km/h')
    s = _normalize_units(s)

    # Remove unwanted special characters but keep allowed punctuation
    # We'll replace any character that is not alnum or allowed punct with a space
    allowed = re.escape(_ALLOWED_PUNCT)
    s = re.sub(rf"[^\w\s{allowed}]", " ", s, flags=re.UNICODE)
    # Remove extra spaces again
    s = re.sub(r"\s+", " ", s).strip()

    # Remove spaces before/after slashes (e.g., '50 km / h' -> '50 km/h')
    s = re.sub(r"\s*/\s*", "/", s)

    # Normalize spacing between number and unit: '50km/h' -> '50 km/h'
    s = re.sub(r"(\d)(km/h|m/s|km/s|%)\b", r"\1 \2", s, flags=re.IGNORECASE)

    # Lower-case the entire input for simpler matching downstream
    s = s.strip()
    return s


if __name__ == '__main__':
    tests = [
        "  50   km / h  ",
        "Tốc độ:  120kmh",
        "giữ khoảng cách!@#",
        "giới hạn: 50km / h; cấm đỗ",
        "tốc độ 30 km s",
    ]
    for t in tests:
        print(repr(t), "->", repr(clean_user_input(t)))
