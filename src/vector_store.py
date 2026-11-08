"""
Vector Store Module for CandidateFit-RAG.

Provides an in-memory vector store optimized for candidate retrieval,
featuring cosine similarity calculation, metadata filtering (by candidate name or section),
and deterministic top-K sorting.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from src.chunking import Chunk


class InMemoryVectorStore:
    """
    Lightweight, highly inspectable vector index.
    Eliminates external vector DB daemon overhead while providing exact cosine similarity.
    """
    def __init__(self):
        self.chunks: List[Chunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self._doc_id_to_index: Dict[str, List[int]] = {}

    def clear(self):
        """Resets the vector store index."""
        self.chunks = []
        self.embeddings = None
        self._doc_id_to_index = {}

    def add_chunks(self, chunks: List[Chunk], embeddings: np.ndarray):
        """
        Adds text chunks and their corresponding normalized dense vector embeddings.
        
        Args:
            chunks: List of Chunk objects.
            embeddings: Numpy array of shape (len(chunks), dimension).
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: received {len(chunks)} chunks but {len(embeddings)} embedding vectors."
            )

        start_idx = len(self.chunks)
        self.chunks.extend(chunks)

        if self.embeddings is None:
            self.embeddings = embeddings.astype(np.float32)
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings.astype(np.float32)])

        # Update index lookup for candidate/document filtering
        for i, chunk in enumerate(chunks):
            idx = start_idx + i
            if chunk.doc_id not in self._doc_id_to_index:
                self._doc_id_to_index[chunk.doc_id] = []
            self._doc_id_to_index[chunk.doc_id].append(idx)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        candidate_name_filter: Optional[str] = None
    ) -> List[Tuple[Chunk, float]]:
        """
        Performs exact nearest-neighbor search via cosine similarity.
        
        Args:
            query_embedding: Normalized query vector of shape (dimension,).
            top_k: Number of highest-similarity chunks to return.
            candidate_name_filter: Optional filter to restrict search to a single candidate.
            
        Returns:
            List of (Chunk, similarity_score) tuples sorted in descending order of score.
        """
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        # Ensure query vector is unit normalized
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            normalized_query = query_embedding / query_norm
        else:
            normalized_query = query_embedding

        # Dot product with pre-normalized chunk embeddings equals cosine similarity
        similarity_scores = np.dot(self.embeddings, normalized_query)

        # Apply candidate filter if requested
        if candidate_name_filter:
            valid_indices = [
                i for i, chunk in enumerate(self.chunks)
                if candidate_name_filter.lower() in chunk.candidate_name.lower()
            ]
        else:
            valid_indices = list(range(len(self.chunks)))

        if not valid_indices:
            return []

        valid_scores = similarity_scores[valid_indices]
        
        # Sort indices by similarity descending
        sorted_sub_indices = np.argsort(valid_scores)[::-1][:top_k]

        results: List[Tuple[Chunk, float]] = []
        for sub_idx in sorted_sub_indices:
            original_idx = valid_indices[sub_idx]
            chunk = self.chunks[original_idx]
            score = float(valid_scores[sub_idx])
            results.append((chunk, score))

        return results

    def get_total_chunks(self) -> int:
        """Returns the total number of indexed chunks."""
        return len(self.chunks)
