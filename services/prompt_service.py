"""Tạo prompt cho trợ lý chuyên về LÝ THUYẾT LÁI XE Ô TÔ tại Việt Nam.

Hàm chính: build_prompt(question, sample_answer=None, reference_text=None)
- question: chuỗi câu hỏi người dùng
- sample_answer: (tuỳ chọn) câu trả lời mẫu để làm ví dụ
- reference_text: (tuỳ chọn) nội dung lý thuyết tham khảo (đã xử lý)

CLI: chạy `python service/promt.py` và làm theo hướng dẫn để nhập câu hỏi.
File này chỉ in prompt ra stdout (không ghi file).
"""

from __future__ import annotations

from typing import Optional


_TEMPLATE = (
    "Bạn là trợ lý AI chuyên về LÝ THUYẾT LÁI XE Ô TÔ tại Việt Nam.\n\n"
    "Nhiệm vụ:\n"
    "- Nhận vào 2 tham số:\n"
    "  1) CÂU HỎI của người dùng.\n"
    "  2) NỘI DUNG LÝ THUYẾT THAM KHẢO (là đoạn text đã được xử lý, trích xuất từ tài liệu chuẩn).\n\n"
    "Quy tắc quan trọng:\n"
    "- Chỉ được dựa vào NỘI DUNG LÝ THUYẾT THAM KHẢO để trả lời.\n"
    "- Dùng NỘI DUNG LÝ THUYẾT THAM KHẢO.\n"
    "- Khi chắc chắn: trả lời đúng trọng tâm, ngắn gọn, rõ ràng; có thể dùng gạch đầu dòng để làm nổi bật ý chính.\n"
    "- Khi NỘI DUNG LÝ THUYẾT THAM KHẢO không đủ để trả lời:\n"
    "  → Hãy trả lời theo phong cách lịch sự – trung lập rằng \"Thông tin chưa đủ để trả lời chính xác. Bạn vui lòng kiểm chứng thêm từ tài liệu gốc hoặc cơ quan có thẩm quyền.\"\n"
    "- Không bịa đặt, không suy đoán ngoài nội dung đã cho.\n"
    "- Văn phong: tiếng Việt, ngắn gọn, trung lập, dễ hiểu.\n\n"
    "Đầu ra: chỉ gồm câu trả lời (không thêm JSON, metadata hay cấu trúc kỹ thuật).\n"
)
    


def build_prompt(question: str, sample_answer: Optional[str] = None, reference_text: Optional[str] = None) -> str:
    """Xây dựng chuỗi prompt cuối cùng từ các tham số đầu vào.

    Hướng dẫn trong prompt yêu cầu trợ lý chỉ trả lời dựa trên `reference_text` (nếu có).
    `sample_answer` là tuỳ chọn để làm ví dụ về cách diễn đạt, nhưng không được dùng để bổ sung thông tin.
    Trả về một chuỗi (str) chứa toàn bộ prompt.
    """
    parts = [_TEMPLATE]
    parts.append(f"CÂU HỎI:\n{question}\n")
    if sample_answer:
        parts.append(f"CÂU TRẢ LỜI MẪU:\n{sample_answer}\n")
    if reference_text:
        parts.append(f"NỘI DUNG LÝ THUYẾT THAM KHẢO:\n{reference_text}\n")

    parts.append("HÃY TRẢ LỜI NGẮN GỌN VÀ DỰA TRÊN NỘI DUNG LÝ THUYẾT THAM KHẢO Ở TRÊN.")
    return "\n".join(parts)


def build_prompt_text(question: str, sample_answer: Optional[str] = None, reference_text: Optional[str] = None) -> str:
    """Alias cho build_prompt — giữ tên rõ ràng rằng hàm trả về TEXT (string).

    Sử dụng khi gọi module từ mã khác: `from service.promt import build_prompt_text`
    Gọi hàm sẽ trả về chuỗi prompt hoàn chỉnh.
    """
    return build_prompt(question, sample_answer=sample_answer, reference_text=reference_text)


def _cli() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Tạo prompt cho trợ lý LÝ THUYẾT LÁI XE Ô TÔ")
    ap.add_argument("--question", type=str, default=None, help="Câu hỏi (nếu muốn chạy không tương tác)")
    ap.add_argument("--sample", type=str, default=None, help="Câu trả lời mẫu (tuỳ chọn)")
    ap.add_argument("--ref-file", type=str, default=None, help="Đường dẫn tới file chứa nội dung tham khảo (tuỳ chọn)")
    args = ap.parse_args()

    q = args.question
    samp = args.sample
    ref = None

    if args.ref_file:
        try:
            ref = open(args.ref_file, "r", encoding="utf-8").read()
        except Exception as e:
            print(f"Không thể đọc file tham khảo: {e}")
            ref = None

    # Nếu không có question từ args -> tương tác với người dùng như trước
    if not q:
        print("Tạo prompt cho trợ lý LÝ THUYẾT LÁI XE Ô TÔ (nhấn Enter để bỏ qua phần tuỳ chọn).")
        q = input("Nhập CÂU HỎI: ").strip()
        if not q:
            print("Không có câu hỏi. Thoát.")
            return
        samp = input("Nhập CÂU TRẢ LỜI MẪU (tuỳ chọn, có thể bỏ): ").strip() or None
        print("Nhập NỘI DUNG LÝ THUYẾT THAM KHẢO (tuỳ chọn). Nhập xong gõ 'END' ở dòng mới:")
        lines = []
        while True:
            line = input()
            if line.strip().upper() == "END":
                break
            lines.append(line)
        ref = "\n".join(lines).strip() or None

    prompt = build_prompt_text(q, sample_answer=samp, reference_text=ref)
    # In prompt ra stdout (dưới dạng text/plain)
    print('\n--- GENERATED PROMPT (text) ---\n')
    print(prompt)
    # Nếu module được gọi từ shell, hàm kết thúc; khi import, gọi build_prompt_text để lấy string


if __name__ == "__main__":
    _cli()
