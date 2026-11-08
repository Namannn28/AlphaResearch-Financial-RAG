"""
Embeddings Engine for Financial Text.
Uses SentenceTransformers ('all-MiniLM-L6-v2') by default with an offline
SVD vectorizer fallback for fast unit testing.
"""

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

        # L2 Normalize
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        return (tfidf / norms).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a single query string."""
        return self.embed_texts([query])[0]
