# tests/providers_demo.py
# -*- coding: utf-8 -*-

def bm25_provider():
    """Yield (ques, ans) cho BM25 test."""
    yield "Biển báo cấm là gì?", "Biển báo cấm hình tròn viền đỏ nền trắng."
    yield "Khoảng cách an toàn là bao nhiêu?", "Tối thiểu 35m khi đi 60 km/h (tham khảo)."
    yield "Xử lý khi gặp tai nạn?", "Dừng xe, bật cảnh báo, đặt biển cảnh báo, gọi cứu hộ."
    yield "Tốc độ tối đa trong khu dân cư?", "Ô tô con tối đa 50 km/h theo quy định."
    yield "Khi đèn vàng nhấp nháy phải làm gì?", "Giảm tốc độ, quan sát kỹ khi qua nút giao."

def semantic_corpus_provider():
    """Yield text cho semantic test (đơn thuần là chuỗi)."""
    yield "Biển báo cấm có hình tròn viền đỏ nền trắng."
    yield "Khoảng cách an toàn phụ thuộc vào tốc độ di chuyển."
    yield "Quy trình xử lý tai nạn giao thông."
    yield "Tốc độ tối đa cho ô tô trong khu dân cư."
    yield "Đèn vàng nhấp nháy thì giảm tốc độ và quan sát."
