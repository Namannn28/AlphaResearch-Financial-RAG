"""
Cross-Encoder Re-Ranking Module for CandidateFit-RAG.

Two-stage retrieval architecture:
1. Stage 1 (Fast Retrieval): Hybrid dense-vector and BM25 search selects top-K chunks.
2. Stage 2 (Precision Re-Ranking): A cross-encoder evaluates (query, chunk) pairs
   jointly with full cross-attention across all transformer layers.
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging

from src.chunking import Chunk
from src.hybrid_retriever import RetrievalResult
from src.config import retrieval_config

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """Encapsulates before-and-after ranking shifts for performance observability."""
    chunk: Chunk
    initial_rank: int
    reranked_rank: int
    initial_score: float
    rerank_score: float
    rank_lift: int  # Positive means the chunk moved up in rank


class CrossEncoderReranker:
    """
    Evaluates query-passage pairs jointly to capture contextual dependencies
    that independent bi-encoder representations can miss.
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", enable_model: bool = True):
        self.model_name = model_name
        self.enable_model = enable_model
        self._model = None
        if self.enable_model:
            self._initialize_model()

    def _initialize_model(self):
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            logger.info("Initialized CrossEncoder reranker: %s", self.model_name)
        except Exception as exc:
            logger.warning("Could not load CrossEncoder (%s). Using Contextual Cross-Scorer fallback. %s", self.model_name, exc)
            self._model = None

    @property
    def is_available(self) -> bool:
        return self._model is not None

    def rerank(
        self,
        query: str,
        initial_results: List[RetrievalResult],
        top_n: int = None
    ) -> List[RerankResult]:
        """
        Re-scores and sorts candidate chunks using cross-attention relevance.
        
        Args:
            query: The question being evaluated.
            initial_results: Stage-1 retrieved chunks.
            top_n: Number of chunks to retain post-reranking.
            
        Returns:
            List of RerankResult objects with before/after ranking diagnostics.
        """
        if not initial_results:
            return []

        top_n = top_n or retrieval_config.rerank_top_n
        chunk_pairs = [[query, res.chunk.text] for res in initial_results]

        if self._model is not None:
            # Model-based cross-attention scoring
            try:
                scores = self._model.predict(chunk_pairs, show_progress_bar=False)
            except Exception as exc:
                logger.error("Error during cross-encoder inference: %s. Using heuristic fallback.", exc)
                scores = self._heuristic_cross_score(query, initial_results)
        else:
            # Heuristic cross-scoring fallback (Entity overlap + semantic salience)
            scores = self._heuristic_cross_score(query, initial_results)

        # Pair scores with initial results
        scored_candidates = []
        for initial_idx, (res, score) in enumerate(zip(initial_results, scores)):
            scored_candidates.append((res, float(score), initial_idx + 1))

        # Sort descending by re-rank score
        scored_candidates.sort(key=lambda item: item[1], reverse=True)

        # Formulate output diagnostics
        reranked_results: List[RerankResult] = []
        for new_idx, (res, rerank_score, orig_rank) in enumerate(scored_candidates[:top_n]):
            new_rank = new_idx + 1
            rank_lift = orig_rank - new_rank  # e.g., rank 4 -> rank 1 gives lift +3
            reranked_results.append(
                RerankResult(
                    chunk=res.chunk,
                    initial_rank=orig_rank,
                    reranked_rank=new_rank,
                    initial_score=res.score,
                    rerank_score=rerank_score,
                    rank_lift=rank_lift
                )
            )

        return reranked_results

    def _heuristic_cross_score(self, query: str, results: List[RetrievalResult]) -> List[float]:
        """
        Calculates joint entity and contextual overlap score when PyTorch cross-encoder is offline.
        Awards bonuses for exact technical matches (acronyms, numbers, metrics) in context.
        Penalizes chunks with zero substantive entity overlap to trigger safety refusal.
        """
        import re
        stopwords = {
            "what", "which", "who", "where", "how", "does", "did", "is", "was", "the",
            "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with", "candidate",
            "experience", "years", "has", "have", "had", "from", "by", "any", "about"
        }
        all_tokens = re.findall(r"\b[A-Za-z0-9\.\-\+]+\b", query.lower())
        substantive_query_words = set([t for t in all_tokens if t not in stopwords and len(t) > 1])
        scores = []

        for res in results:
            text = res.chunk.text.lower()
            if not substantive_query_words:
                scores.append(res.score)
                continue

            overlap_words = [word for word in substantive_query_words if word in text]
            overlap_count = len(overlap_words)
            overlap_ratio = overlap_count / len(substantive_query_words)

            # If none of the substantive query keywords appear in the retrieved text,
            # clamp score to near zero to ensure out-of-corpus refusal guardrail triggers.
            if overlap_count == 0:
                scores.append(0.05)
                continue

            # Metric detection bonus (numbers like '45%', '18,000', '28ms')
            metric_bonus = 0.15 if any(char.isdigit() for char in query) and any(char.isdigit() for char in text) else 0.0

            # Header match bonus
            header_bonus = 0.10 if any(w in res.chunk.section_title.lower() for w in overlap_words) else 0.0

            combined_score = 0.4 * res.score + 0.5 * overlap_ratio + metric_bonus + header_bonus
            scores.append(combined_score)

        return scores

        return scores
