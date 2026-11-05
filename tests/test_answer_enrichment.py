# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) in sys.path:
    sys.path.remove(str(TESTS_DIR))
PACKAGE_ROOT = TESTS_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from retriever.api_server import (  # noqa: E402
    _enrich_short_answer,
    _marks_reasoned_sections,
    build_simple_vietnamese_prompt,
)


@pytest.mark.parametrize(
    "verbosity, expected",
    [
        ("minimal", "Trả lời ngắn gọn"),
        ("normal", "Có thể suy luận mở rộng"),
        ("detailed", "Suy luận hợp lý"),
    ],
)
def test_build_prompt_contains_detail_guidance(verbosity, expected):
    prompt = build_simple_vietnamese_prompt(
        "Người lái xe cần làm gì?",
        "[1]\nGiữ khoảng cách an toàn với xe phía trước.",
        verbosity=verbosity,
    )
    assert expected in prompt
    assert "Không khẳng định thông tin trái với tài liệu" in prompt
    assert "Suy luận hợp lý" in prompt


def test_enrich_short_answer_appends_reference():
    context = (
        "[1]\nKhông được quay đầu xe tại nơi có vạch kẻ liền hai bên đường."
        "\n\n[2]\nTrường hợp cần quay đầu phải tìm vị trí cho phép và đảm bảo an toàn."
    )
    enriched = _enrich_short_answer("Không", context, max_points=2)
    lines = [ln.strip() for ln in enriched.splitlines() if ln.strip()]
    assert any("tham khảo [1]" in ln for ln in lines)
    assert any(ln.startswith("-") for ln in lines)


def test_marks_reasoned_sections_detection():
    assert _marks_reasoned_sections("Suy luận hợp lý: áp dụng trong tình huống mưa lớn")
    assert _marks_reasoned_sections("Ví dụ minh họa (tham khảo [2])")
    assert not _marks_reasoned_sections("Câu trả lời không có nhãn suy luận")
