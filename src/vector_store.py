"""
In-Memory Dense Vector Store with Metadata Filtering and Cosine Similarity.
Designed for fast local execution without heavyweight external server dependencies.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from src.chunking import FinancialChunk

class FinancialVectorStore:
    """Stores embedded financial chunks and performs cosine nearest-neighbor search."""

    def __init__(self):
        self.chunks: List[FinancialChunk] = []
        self.embeddings: Optional[np.ndarray] = None

    def add_chunks(self, chunks: List[FinancialChunk], embeddings: np.ndarray):
        """Adds chunks and their corresponding embedding vectors to the store."""
        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunk count {len(chunks)} does not match embedding count {len(embeddings)}")

        self.chunks.extend(chunks)
        if self.embeddings is None:
            self.embeddings = embeddings.astype(np.float32)
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings.astype(np.float32)])

    def similarity_search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filter_company: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Calculates cosine similarities between query vector and stored document vectors.
        Returns top-k most similar chunks with metadata.
        """
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        # Cosine similarity for normalized vectors is simply the dot product
        scores = np.dot(self.embeddings, query_vector)

        ranked_indices = np.argsort(scores)[::-1]
        results: List[Dict[str, Any]] = []

        for idx in ranked_indices:
            chunk = self.chunks[idx]
            # Optional company filter
            if filter_company and filter_company.lower() not in chunk.company.lower() and filter_company.upper() != chunk.ticker:
                continue

            results.append({
                "chunk": chunk,
                "score": float(scores[idx]),
                "rank": len(results) + 1
            })

            if len(results) >= top_k:
                break

        return results

    def clear(self):
        """Resets the vector store."""
        self.chunks = []
        self.embeddings = None
