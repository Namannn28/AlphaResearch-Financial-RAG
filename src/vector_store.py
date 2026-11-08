r"""
==============================================================================
STEP 3B: IN-MEMORY VECTOR STORE (src/vector_store.py)
==============================================================================
Role in RAG Pipeline:
    Maintains an in-memory index of embedded document chunks, metadata, and
    performs fast cosine nearest-neighbor search with optional metadata filtering.

Key Concepts for Interviews:
    - Zero external infrastructure: Runs locally without requiring Docker,
      Chroma server, or Pinecone credentials.
    - Vectorized Cosine Search: Uses NumPy matrix multiplication:
      $\mathbf{S} = \mathbf{E} \cdot \mathbf{q}$
      for sub-millisecond retrieval over thousands of chunks.
    - Dynamic Indexing: Supports live document additions on the fly when users
      drag and drop new PDFs.

Run directly to test:
    python src/vector_store.py
==============================================================================
"""

import sys
from pathlib import Path

# Add project root to path for direct module execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import List, Dict, Any, Optional
import numpy as np
from src.chunking import FinancialChunk, SectionAwareFinancialChunker
from src.ingestion import DocumentIngestionEngine
from src.embeddings import FinancialEmbeddingEngine
from src.config import REPORTS_DIR


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


if __name__ == "__main__":
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 70)
    print("[STEP 3B DEMO] Vector Store & Semantic Search")
    print("=" * 70)
    docs = DocumentIngestionEngine().load_all_reports(REPORTS_DIR)
    chunks = []
    chunker = SectionAwareFinancialChunker()
    for d in docs:
        chunks.extend(chunker.chunk_document(d))

    embedder = FinancialEmbeddingEngine()
    vectors = embedder.embed_texts([c.text for c in chunks])

    store = FinancialVectorStore()
    store.add_chunks(chunks, vectors)
    print(f"Indexed {len(chunks)} chunks into vector store. Stored matrix shape: {store.embeddings.shape}")

    test_q = "What was Tesla's vehicle delivery count and gross margin?"
    q_vec = embedder.embed_query(test_q)
    hits = store.similarity_search(q_vec, top_k=3)

    print(f"\nTop 3 Vector Matches for: \"{test_q}\"\n")
    for hit in hits:
        c = hit["chunk"]
        print(f"  [Score: {hit['score']:.4f}] {c.company} ({c.ticker}) | {c.section_title}")
        print(f"    Excerpt: {c.text.splitlines()[1][:120]}...\n")

    print("[OK] Vector store module operational.")
