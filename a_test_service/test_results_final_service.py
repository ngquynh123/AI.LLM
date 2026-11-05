import sys, os
import json

# ensure project root (parent of services/) is on sys.path so `from services import ...` works
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from services import results_final_service as rfs

QUERIES = [
    "ô tô được phân loại theo gì",
    "hi",
    "nguyên nhân nào sau đây có thể dẫn đến cháy nổ trên ô tô điện",
    "đạo đưc là gì"
]

if __name__ == '__main__':
    for q in QUERIES:
        out = rfs.run_text(q, top_k=5)
        print('---')
        print('Query:', q)
        print(json.dumps(out, ensure_ascii=False, indent=2))
