r"""
==============================================================================
STEP 4B: HYBRID RETRIEVER & RECIPROCAL RANK FUSION (src/hybrid_retriever.py)
==============================================================================
Role in RAG Pipeline:
    Fuses semantic hits from the Dense Vector Store and keyword hits from the
    BM25 Lexical Retriever into a single reranked candidate list using
    Reciprocal Rank Fusion (RRF).

Key Architectural & Design Principles:
    - Why Hybrid Retrieval: Combines semantic generalization (Dense) with exact
      number/ticker precision (BM25), eliminating blind spots of both approaches.
    - Reciprocal Rank Fusion (RRF):
      $$\text{Score}_{\text{RRF}}(d) = \sum_{m \in \{\text{Dense}, \text{BM25}\}} \frac{w_m}{k + \text{rank}_m(d)}$$
      where $k=60$ is the standard smoothing constant.
    - Score Invariance: RRF works on ordinal ranks rather than raw cosine / BM25
      scores, completely bypassing the need for score normalization or calibration.

Run directly to test:
    python src/hybrid_retriever.py
==============================================================================
"""

import sys
from pathlib import Path

# Add project root to path for direct module execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import List, Dict, Any, Optional
from src.chunking import FinancialChunk, SectionAwareFinancialChunker
from src.ingestion import DocumentIngestionEngine
from src.vector_store import FinancialVectorStore
from src.lexical_search import BM25OkapiRetriever
from src.embeddings import FinancialEmbeddingEngine
from src.config import REPORTS_DIR


class HybridFinancialRetriever:
    """
    Executes hybrid retrieval:
    1. Dense cosine similarity via sentence embeddings (captures semantics & paraphrasing).
    2. BM25Okapi keyword search (captures exact metrics, numbers, and ticker symbols).
    3. Reciprocal Rank Fusion (RRF) to merge candidate lists into an unified ranking.
    """

    def __init__(
        self,
        vector_store: FinancialVectorStore,
        bm25_retriever: BM25OkapiRetriever,
        embedding_engine: FinancialEmbeddingEngine,
        rrf_constant: int = 60,
        dense_weight: float = 0.5,
        lexical_weight: float = 0.5
    ):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.embedding_engine = embedding_engine
        self.rrf_constant = rrf_constant
        self.dense_weight = dense_weight
        self.lexical_weight = lexical_weight

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filter_company: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Performs hybrid retrieval and returns top fused candidates."""
        # 1. Dense Vector Search
        query_vector = self.embedding_engine.embed_query(query)
        dense_hits = self.vector_store.similarity_search(
            query_vector,
            top_k=top_k * 2,
            filter_company=filter_company
        )

        # 2. BM25 Keyword Search
        bm25_hits = self.bm25_retriever.search(
            query,
            top_k=top_k * 2,
            filter_company=filter_company
        )

        # 3. Reciprocal Rank Fusion (RRF)
        fused_scores: Dict[str, float] = {}
        chunk_map: Dict[str, FinancialChunk] = {}
        dense_ranks: Dict[str, int] = {}
        bm25_ranks: Dict[str, int] = {}

        for rank_idx, hit in enumerate(dense_hits, start=1):
            cid = hit["chunk"].chunk_id
            chunk_map[cid] = hit["chunk"]
            dense_ranks[cid] = rank_idx
            fused_scores[cid] = fused_scores.get(cid, 0.0) + (self.dense_weight / (self.rrf_constant + rank_idx))

        for rank_idx, hit in enumerate(bm25_hits, start=1):
            cid = hit["chunk"].chunk_id
            chunk_map[cid] = hit["chunk"]
            bm25_ranks[cid] = rank_idx
            fused_scores[cid] = fused_scores.get(cid, 0.0) + (self.lexical_weight / (self.rrf_constant + rank_idx))

        # Sort by fused score descending
        sorted_chunks = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

        results: List[Dict[str, Any]] = []
        for cid, score in sorted_chunks[:top_k]:
            results.append({
                "chunk": chunk_map[cid],
                "score": score,
                "dense_rank": dense_ranks.get(cid, None),
                "bm25_rank": bm25_ranks.get(cid, None)
            })

        return results


if __name__ == "__main__":
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 70)
    print("[STEP 4B DEMO] Hybrid Retrieval (Dense + BM25 Fusion)")
    print("=" * 70)
    docs = DocumentIngestionEngine().load_all_reports(REPORTS_DIR)
    chunks = []
    chunker = SectionAwareFinancialChunker()
    for d in docs:
        chunks.extend(chunker.chunk_document(d))

    embedder = FinancialEmbeddingEngine()
    vectors = embedder.embed_texts([c.text for c in chunks])

    vstore = FinancialVectorStore()
    vstore.add_chunks(chunks, vectors)

    bm25 = BM25OkapiRetriever()
    bm25.index_chunks(chunks)

    hybrid = HybridFinancialRetriever(vstore, bm25, embedder)
    test_q = "What was Amazon's AWS segment net sales and operating income?"
    hits = hybrid.retrieve(test_q, top_k=3)

    print(f"Hybrid Results for: \"{test_q}\"\n")
    for idx, hit in enumerate(hits, start=1):
        c = hit["chunk"]
        print(f"  #{idx} [RRF Score: {hit['score']:.5f}] (Dense Rank: {hit['dense_rank']}, BM25 Rank: {hit['bm25_rank']})")
        print(f"      {c.company} ({c.ticker}) | {c.section_title}")
        print(f"      Excerpt: {c.text.splitlines()[1][:110]}...\n")

    print("[OK] Hybrid retrieval module operational.")
