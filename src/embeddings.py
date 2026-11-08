"""
Embeddings Module for CandidateFit-RAG.

This module provides vector embedding generation with clear trade-off justifications.
Supports:
1. SentenceTransformersEmbedder (Primary):
   - Model: 'all-MiniLM-L6-v2' (384-dimensional dense vectors)
   - Trade-off Rationale:
     * Latency: Extremely lightweight (~15ms per batch on standard CPU).
     * Dimension: 384 dimensions vs 1536 (OpenAI text-embedding-3-small) reduces
       RAM footprint by 75% while maintaining competitive MTEB retrieval accuracy.
     * Cost & Privacy: Runs 100% locally with zero cloud API token costs and ensures
       candidate PII (Personally Identifiable Information) never leaves the host.
     * License: Permissive Apache 2.0 open-source license.

2. FallbackDenseEmbedder:
   - Zero-dependency dense embedding generator using TF-IDF + Truncated SVD (LSA)
     with L2 unit normalization to ensure offline predictability without heavy downloads.
"""

from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np
import logging

from src.config import retrieval_config

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """Abstract interface for all dense embedding implementations."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns the embedding dimensionality."""
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Computes dense vectors for a list of document chunks. Shape: (N, D)."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Computes dense vector for a single search query. Shape: (D,)."""
        pass


class SentenceTransformersEmbedder(BaseEmbedder):
    """
    Primary dense embedding engine powered by Hugging Face sentence-transformers.
    """
    def __init__(self, model_name: str = None):
        self.model_name = model_name or retrieval_config.dense_model_name
        self._model = None
        self._dim = retrieval_config.dense_dimension
        self._initialize_model()

    def _initialize_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            # Load model onto CPU with quiet logging
            self._model = SentenceTransformer(self.model_name)
            if hasattr(self._model, "get_embedding_dimension"):
                self._dim = self._model.get_embedding_dimension()
            else:
                self._dim = self._model.get_sentence_embedding_dimension()
            logger.info("Initialized SentenceTransformer model: %s (dim=%d)", self.model_name, self._dim)
        except Exception as exc:
            logger.warning(
                "Could not load SentenceTransformer (%s). Falling back to Fast Dense SVD Embedder. Error: %s",
                self.model_name,
                exc
            )
            self._model = None

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def is_available(self) -> bool:
        return self._model is not None

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        if self._model is not None:
            embeddings = self._model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            return np.array(embeddings, dtype=np.float32)
        else:
            raise RuntimeError("SentenceTransformer model is not loaded.")

    def embed_query(self, query: str) -> np.ndarray:
        if self._model is not None:
            embedding = self._model.encode(
                query,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            return np.array(embedding, dtype=np.float32)
        else:
            raise RuntimeError("SentenceTransformer model is not loaded.")


class FallbackDenseEmbedder(BaseEmbedder):
    """
    Lightweight, deterministic local dense embedder utilizing Scikit-learn
    TF-IDF with sublinear term scaling and Truncated SVD dimensionality reduction.
    Guarantees that the RAG pipeline operates reliably even on resource-constrained
    or fully air-gapped systems.
    """
    def __init__(self, dimension: int = 128):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        self._dim = dimension
        self.vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            ngram_range=(1, 2),
            stop_words="english",
            max_features=2500
        )
        self.svd = None
        self.is_fitted = False

    @property
    def dimension(self) -> int:
        return self._dim

    def fit_corpus(self, texts: List[str]):
        """Fits vocabulary and latent decomposition components across candidate corpus."""
        from sklearn.decomposition import TruncatedSVD

        tfidf_matrix = self.vectorizer.fit_transform(texts)
        actual_dim = min(self._dim, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
        actual_dim = max(8, actual_dim)
        self._dim = actual_dim

        self.svd = TruncatedSVD(n_components=actual_dim, random_state=42)
        self.svd.fit(tfidf_matrix)
        self.is_fitted = True

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return (vectors / norms).astype(np.float32)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not self.is_fitted:
            self.fit_corpus(texts)

        tfidf_matrix = self.vectorizer.transform(texts)
        dense_vectors = self.svd.transform(tfidf_matrix)
        return self._normalize(dense_vectors)

    def embed_query(self, query: str) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("FallbackDenseEmbedder must be fit on candidate corpus before embedding queries.")

        tfidf_vec = self.vectorizer.transform([query])
        dense_vec = self.svd.transform(tfidf_vec)
        normalized = self._normalize(dense_vec)
        return normalized[0]


def get_embedder(prefer_sentence_transformer: bool = True) -> BaseEmbedder:
    """
    Factory function returning the most capable available embedding model.
    Tries SentenceTransformers first; falls back smoothly if not installed/configured.
    """
    if prefer_sentence_transformer:
        st_embedder = SentenceTransformersEmbedder()
        if st_embedder.is_available:
            return st_embedder

    logger.info("Using FallbackDenseEmbedder.")
    return FallbackDenseEmbedder(dimension=retrieval_config.dense_dimension)
