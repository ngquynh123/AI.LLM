
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer

def normalize(v: np.ndarray) -> np.ndarray:
    if v.ndim == 1:
        n = np.linalg.norm(v);  return v if n==0 else v/n
    n = np.linalg.norm(v, axis=1, keepdims=True); n[n==0]=1.0
    return v / n

class SemanticSearcher:
    def __init__(self, index: np.ndarray, model_id: str="AITeamVN/Vietnamese_Embedding", device: str="cpu"):
        self.model = SentenceTransformer(model_id, device=device)
        self.corpus = normalize(index.astype(np.float32))

    def search(self, query: str, topk: int=8) -> List[Dict[str, float]]:
        q = self.model.encode([query], batch_size=1, normalize_embeddings=True)[0].astype(np.float32)
        sims = np.dot(self.corpus, q)   # cosine vì đã normalize
        k = int(min(topk, len(sims)))
        top = np.argpartition(-sims, k-1)[:k]
        top = top[np.argsort(-sims[top])]
        return [{"rid": int(i), "score": float(sims[i])} for i in top]
