"""
Hybrid Retrieval Module for CandidateFit-RAG.

Combines Dense Semantic Vector Search with BM25 Lexical Search using
Reciprocal Rank Fusion (RRF). RRF provides score-scale invariance, avoiding
the need to calibrate disparate cosine similarity [-1, 1] and BM25 [0, inf) distributions.
"""

from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import time

from src.chunking import Chunk
from src.embeddings import BaseEmbedder
from src.vector_store import InMemoryVectorStore
from src.lexical_search import BM25LexicalSearcher
from src.config import retrieval_config


@dataclass
class RetrievalResult:
    """Detailed output of a retrieval operation with diagnostic trace."""
    chunk: Chunk
    score: float
    dense_rank: Optional[int] = None
    lexical_rank: Optional[int] = None
    retrieval_method: str = "hybrid"


class HybridRetriever:
    """
    Fuses dense vector representations and BM25 keyword matching.
    """
    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: InMemoryVectorStore,
        lexical_searcher: BM25LexicalSearcher,
        rrf_constant: int = None
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.lexical_searcher = lexical_searcher
        self.rrf_k = rrf_constant or retrieval_config.rrf_constant

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
        candidate_name_filter: Optional[str] = None
    ) -> List[RetrievalResult]:
        """
        Executes query retrieval across the indexed corpus.
        
        Args:
            query: The user or recruiter question.
            top_k: Number of candidate chunks to return.
            mode: 'hybrid' (default), 'dense_only', or 'lexical_only'.
            candidate_name_filter: Optional candidate name constraint.
            
        Returns:
            List of RetrievalResult objects sorted by relevance score descending.
        """
        if mode == "dense_only":
            return self._retrieve_dense(query, top_k, candidate_name_filter)
        elif mode == "lexical_only":
            return self._retrieve_lexical(query, top_k, candidate_name_filter)
        elif mode == "hybrid":
            return self._retrieve_hybrid(query, top_k, candidate_name_filter)
        else:
            raise ValueError(f"Unknown retrieval mode: {mode}")

    def _retrieve_dense(
        self,
        query: str,
        top_k: int,
        candidate_name_filter: Optional[str]
    ) -> List[RetrievalResult]:
        query_embedding = self.embedder.embed_query(query)
        dense_hits = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            candidate_name_filter=candidate_name_filter
        )
        return [
            RetrievalResult(
                chunk=chunk,
                score=score,
                dense_rank=rank + 1,
                lexical_rank=None,
                retrieval_method="dense_only"
            )
            for rank, (chunk, score) in enumerate(dense_hits)
        ]

    def _retrieve_lexical(
        self,
        query: str,
        top_k: int,
        candidate_name_filter: Optional[str]
    ) -> List[RetrievalResult]:
        lexical_hits = self.lexical_searcher.search(
            query=query,
            top_k=top_k,
            candidate_name_filter=candidate_name_filter
        )
        return [
            RetrievalResult(
                chunk=chunk,
                score=score,
                dense_rank=None,
                lexical_rank=rank + 1,
                retrieval_method="lexical_only"
            )
            for rank, (chunk, score) in enumerate(lexical_hits)
        ]

    def _retrieve_hybrid(
        self,
        query: str,
        top_k: int,
        candidate_name_filter: Optional[str]
    ) -> List[RetrievalResult]:
        # Step 1: Query dense vector store
        query_embedding = self.embedder.embed_query(query)
        dense_hits = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k * 2,
            candidate_name_filter=candidate_name_filter
        )

        # Step 2: Query BM25 lexical ranker
        lexical_hits = self.lexical_searcher.search(
            query=query,
            top_k=top_k * 2,
            candidate_name_filter=candidate_name_filter
        )

        # Step 3: Reciprocal Rank Fusion (RRF)
        # RRF_Score(doc) = 1 / (K + rank_dense) + 1 / (K + rank_bm25)
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Chunk] = {}
        dense_ranks: Dict[str, int] = {}
        lexical_ranks: Dict[str, int] = {}

        for rank, (chunk, _) in enumerate(dense_hits):
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            dense_ranks[cid] = rank + 1
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + (rank + 1)))

        for rank, (chunk, _) in enumerate(lexical_hits):
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            lexical_ranks[cid] = rank + 1
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + (rank + 1)))

        # Sort candidate chunks by combined RRF score
        sorted_chunks = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

        return [
            RetrievalResult(
                chunk=chunk_map[cid],
                score=score,
                dense_rank=dense_ranks.get(cid),
                lexical_rank=lexical_ranks.get(cid),
                retrieval_method="hybrid_rrf"
            )
            for cid, score in sorted_chunks
        ]
