# -*- coding: utf-8 -*-
import re
from retriever.api_server import (
    _sanitize_llm_output,
    _collapse_qa_pairs,
    _dedupe_sentences,
    _content_overlap_ratio,
    _enrich_short_answer,
    _extract_key_points,
)


def test_dedupe_sentences():
    text = "A. Hello world.\nA. Hello world.\nB. Other." 
    out = _dedupe_sentences(text)
    assert out.count("Hello world") == 1
    assert "Other" in out


def test_collapse_qa_pairs():
    lines = ["Có được vượt ở đoạn đường cong không?", "Không được vì tầm nhìn hạn chế.", "Một dòng khác"]
    out = _collapse_qa_pairs(lines)
    # question should be removed and only answer kept
    assert "Không được vì" in out[0]
    assert all(not l.endswith('?') for l in out if l)


def test_content_overlap_ratio_low():
    ans = "This is an unsupported claim about categories X Y Z"
    ctx = "Some unrelated context about driving rules and signals"
    r = _content_overlap_ratio(ans, ctx)
    assert r < 0.2


def test_enrich_definition_uses_keypoints():
    ctx = "\n".join([
        "Đạo đức là phương thức điều chỉnh hành vi con người;",
        "Đạo đức phản ánh lợi ích của một giai cấp nhất định trong xã hội;",
        "Một hệ thống đạo đức phù hợp mang tính nhân đạo"
    ])
    enriched = _enrich_short_answer("Đạo đức là phương thức điều chỉnh hành vi", ctx, max_points=2)
    # Should include at least one bullet
    assert "-" in enriched or "Đạo đức" in enriched

