"""
Service wrapper for lesson embedding using the SentenceTransformer model
- exposes: embed_texts(texts), embed_documents(documents)
- designed to be lightweight and importable from tests or CLI runners
"""
from pathlib import Path
import json
from typing import List, Any, Dict, Optional
import numpy as np
import hashlib

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    SentenceTransformer = None


DEFAULT_MODEL_ID = "AITeamVN/Vietnamese_Embedding"


class EmbeddingService:
    def __init__(self, model_id: str = DEFAULT_MODEL_ID, use_dummy: bool = False, dim: int = 768):
        """If use_dummy is True, generate deterministic hash-based embeddings of length `dim`.
        Otherwise try to load SentenceTransformer and raise if not available.
        """
        self.dim = int(dim)
        if use_dummy:
            self.model = None
        else:
            if SentenceTransformer is None:
                raise RuntimeError("sentence_transformers not available in this environment; pass use_dummy=True to run without the model")
            self.model = SentenceTransformer(model_id)

    def embed_texts(self, texts: List[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        """Encode a list of texts to embeddings (float32 numpy array).

        Returns shape (N, D) numpy array.
        """
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        if self.model is None:
            # deterministic per-text hash -> seed -> random vector
            def _vec(t: str) -> np.ndarray:
                h = hashlib.sha256(t.encode("utf-8")).digest()
                seed = int.from_bytes(h[:8], "little") & 0xFFFFFFFF
                rng = np.random.RandomState(seed)
                v = rng.randn(self.dim).astype(np.float32)
                # normalize for cosine-friendly behaviour
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

    def embed_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Embed documents that contain a "chunks" list of {"text": ...} entries.
        The function will write an "embed" field (list) into each chunk with valid text.
        """
        # collect texts and references
        refs = []  # (doc_idx, chunk_idx)
        texts = []
        for di, doc in enumerate(documents):
            for ci, chunk in enumerate(doc.get("chunks", [])):
                txt = (chunk.get("text") or "").strip()
                if txt:
                    refs.append((di, ci))
                    texts.append(txt)

        if texts:
            embs = self.embed_texts(texts)
            for (di, ci), emb in zip(refs, embs):
                documents[di]["chunks"][ci]["embed"] = emb.tolist()

        # ensure chunks without text have embed=None
        for doc in documents:
            for chunk in doc.get("chunks", []):
                if "embed" not in chunk:
                    chunk["embed"] = None

        return documents


# Helper CLI runner similar to the original embed/em_lesson.py
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python em_lesson_service.py <input_json_path>")
        sys.exit(1)
    in_path = Path(sys.argv[1])
    if not in_path.exists():
        print(f"Input file not found: {in_path}")
        sys.exit(1)
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    svc = EmbeddingService()
    out = svc.embed_documents(data)
    out_dir = Path("embed_data/lesson")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{in_path.stem}_with_emb.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Wrote {out_path}")