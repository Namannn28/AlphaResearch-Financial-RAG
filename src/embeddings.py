r"""
==============================================================================
STEP 3A: EMBEDDINGS ENGINE (src/embeddings.py)
==============================================================================
Role in RAG Pipeline:
    Converts unstructured textual chunks and user queries into dense 384-dimensional
    vector representations that capture semantic meaning and conceptual similarity.

Key Architectural & Design Principles:
    - Model: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, 80MB).
    - L2 Normalization: All vectors are normalized to unit length ($\|v\| = 1$).
      This converts expensive cosine similarity into a single fast dot product:
      $\cos(\theta) = \mathbf{u} \cdot \mathbf{v}$.
    - Semantic Generalization: Maps paraphrases to nearby coordinates in vector
      space (e.g., "money spent on cloud hardware" -> "capital expenditures").

Run directly to test:
    python src/embeddings.py
==============================================================================
"""

import sys
from pathlib import Path

# Add project root to path for direct module execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import List
import numpy as np


class FinancialEmbeddingEngine:
    """Computes dense normalized vector representations for queries and documents."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", prefer_sentence_transformer: bool = True):
        self.model_name = model_name
        self.backend = None
        self.dimension = 384

        if prefer_sentence_transformer:
            try:
                from sentence_transformers import SentenceTransformer
                self.backend = SentenceTransformer(model_name)
                self.is_neural = True
            except Exception:
                self.backend = None
                self.is_neural = False
        else:
            self.is_neural = False

        if self.backend is None:
            self._init_fallback()

    def _init_fallback(self):
        """Initializes a lightweight TF-IDF fallback vectorizer."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(max_features=self.dimension, stop_words="english")
        self.is_neural = False
        self._fitted = False

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embeds a list of text strings into normalized vectors."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        if self.is_neural and self.backend is not None:
            embeddings = self.backend.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return embeddings.astype(np.float32)

        # Fallback path
        if not getattr(self, "_fitted", False):
            tfidf = self.vectorizer.fit_transform(texts).toarray()
            self._fitted = True
        else:
            tfidf = self.vectorizer.transform(texts).toarray()

        if tfidf.shape[1] < self.dimension:
            padded = np.zeros((tfidf.shape[0], self.dimension), dtype=np.float32)
            padded[:, :tfidf.shape[1]] = tfidf
            tfidf = padded
        else:
            tfidf = tfidf[:, :self.dimension]

        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return (tfidf / norms).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a single query string into a 1D normalized vector."""
        vectors = self.embed_texts([query])
        return vectors[0]


if __name__ == "__main__":
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 70)
    print("[STEP 3A DEMO] SentenceTransformer Vector Embeddings")
    print("=" * 70)
    engine = FinancialEmbeddingEngine()
    print(f"Backend: {'Neural SentenceTransformer (' + engine.model_name + ')' if engine.is_neural else 'TF-IDF Fallback'}")
    print(f"Embedding Dimension: {engine.dimension}")

    samples = [
        "Apple's total net sales reached $383 billion in fiscal 2023.",
        "iPhone revenue accounted for over 50% of Apple's total annual turnover.",
        "Amazon Web Services AWS generated $90.7 billion in revenue.",
        "Tesla delivered over 1.8 million electric vehicles worldwide."
    ]
    vectors = engine.embed_texts(samples)
    print(f"Encoded {len(samples)} sample texts into matrix of shape: {vectors.shape}")

    q = "How much did Apple make from iPhone sales?"
    q_vec = engine.embed_query(q)
    print(f"Query: \"{q}\" -> Vector shape: {q_vec.shape}")

    similarities = np.dot(vectors, q_vec)
    print("\nCosine Similarities to Query:")
    for text, score in zip(samples, similarities):
        print(f"  [{score:.4f}] {text}")

    print("\n[OK] Embeddings module operational.")
