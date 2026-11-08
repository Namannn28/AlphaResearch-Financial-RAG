"""
Hybrid Retriever combining Dense Vector and BM25 Lexical Search
using Reciprocal Rank Fusion (RRF).
"""

from typing import List, Dict, Any, Optional
from src.chunking import FinancialChunk
from src.vector_store import FinancialVectorStore
from src.lexical_search import BM25OkapiRetriever
from src.embeddings import FinancialEmbeddingEngine

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
        # RRF formula: score = w / (k + rank)
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
