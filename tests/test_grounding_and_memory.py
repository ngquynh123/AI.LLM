# -*- coding: utf-8 -*-
import re
from retriever.api_server import (
    _sanitize_llm_output,
    _ground_if_contradiction,
    _is_yes_no_question,
    _resolve_followup_with_memory,
    _update_conv_memory,
    _get_sess_id,
    STATE,
)


def test_sanitize_llm_output_removes_exam_artifacts():
    raw = """
[B4]
Q: Bạn đã từng lái xe sau khi uống rượu bia chưa?
A) Chưa bao giờ.
B) Đã từng lái xe sau khi uống rượu bia nhưng không gây tai nạn.
- Một dòng hữu ích
""".strip()
    out = _sanitize_llm_output(raw)
    # Should remove [B4] and Q:/A) lines, keep the bullet
    assert "[B4]" not in out
    assert re.search(r"^Q:\s*", out, flags=re.IGNORECASE) is None
    assert re.search(r"^[A-D]\)\s", out, flags=re.MULTILINE) is None
    assert "Một dòng hữu ích" in out


def test_grounding_polarity_rewrite():
    # Context prohibiting action; answer incorrectly says it's allowed
    ctx = "Hành vi điều khiển phương tiện khi có nồng độ cồn bị nghiêm cấm. Tuyệt đối cấm lái xe khi đã uống rượu bia."
    ans = "Có, bạn vẫn có thể lái xe nếu cảm thấy tỉnh táo."
    g = _ground_if_contradiction(ans, ctx, "Sau khi uống rượu bia, có thể lái xe không?")
    assert g.get("grounded") is True
    assert g.get("text", "").splitlines()[0].strip().lower().startswith("không")


def test_yes_no_detection():
    assert _is_yes_no_question("Tôi có được quay đầu xe ở đây không?") is True
    assert _is_yes_no_question("Có thể rẽ trái không?") is True
    assert _is_yes_no_question("Trình bày khái niệm khoảng cách an toàn") is False


def test_followup_resolution_why():
    # Seed memory
    sid = _get_sess_id("pytest", "api")
    STATE["conv"][sid] = {"last_q": "Tôi có được quay đầu xe ở đoạn này hay không?", "last_a": "Không"}
    out = _resolve_followup_with_memory("tại sao không", sid)
    # Should include lý do template pointing back to previous question
    assert out.lower().startswith("tại sao")
    assert "quay đầu xe" in out
