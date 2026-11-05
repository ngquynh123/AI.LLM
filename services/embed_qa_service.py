"""
Service wrapper that mirrors the logic in `embed/test_embed_qa.py`.
Provides:
- embed_texts(texts)
- build_qa_text(item)
- embed_qa_items(items)

The service is importable and usable from tests or small runners.
"""
from pathlib import Path
import json
from typing import List, Dict, Any
import numpy as np
import hashlib

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

DEFAULT_MODEL_ID = "AITeamVN/Vietnamese_Embedding"


class EmbedQAService:
    def __init__(self, model_id: str = DEFAULT_MODEL_ID, use_dummy: bool = False, dim: int = 768):
        self.dim = int(dim)
        if use_dummy:
            self.model = None
        else:
            if SentenceTransformer is None:
                raise RuntimeError("sentence_transformers not installed in this environment; pass use_dummy=True to run without the model")
            self.model = SentenceTransformer(model_id)

    def embed_texts(self, texts: List[str], batch_size: int = 64, normalize: bool = True) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        if self.model is None:
            def _vec(t: str) -> np.ndarray:
                h = hashlib.sha256(t.encode("utf-8")).digest()
                seed = int.from_bytes(h[:8], "little") & 0xFFFFFFFF
                rng = np.random.RandomState(seed)
                v = rng.randn(self.dim).astype(np.float32)
                norm = np.linalg.norm(v)
                if norm > 0:
                    v /= norm
                return v

            embs = np.stack([_vec(t) for t in texts], axis=0)
            return embs
        embs = self.model.encode(
            texts,
            normalize_embeddings=normalize,
            batch_size=batch_size,
            show_progress_bar=False,
        ).astype(np.float32)
        return embs

    @staticmethod
    def build_qa_text(item: Dict[str, Any]) -> str:
        q = (item.get("question") or "").strip()
        a = (item.get("answer") or "").strip()
        if q and a:
            return f"Q: {q}\nA: {a}"
        return q or a

    def embed_qa_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        idx_refs = []
        texts = []
        for i, item in enumerate(items):
            text = self.build_qa_text(item)
            if text:
                idx_refs.append(i)
                texts.append(text)

        if texts:
            embs = self.embed_texts(texts)
            for i, emb in zip(idx_refs, embs):
                items[i]["embed"] = emb.tolist()

        for i, item in enumerate(items):
            if "embed" not in item:
                item["embed"] = None

        return items


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python embed_qa_service.py <path_to_qa_json>")
        sys.exit(1)
    in_path = Path(sys.argv[1])
    if not in_path.exists():
        print(f"Input file not found: {in_path}")
        sys.exit(1)
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    svc = EmbedQAService()
    out = svc.embed_qa_items(data)
    out_dir = Path("embed_data/dialogue")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{in_path.stem}_with_emb.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Wrote {out_path}")