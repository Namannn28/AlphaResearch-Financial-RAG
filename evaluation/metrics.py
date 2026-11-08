"""
Evaluation Metrics Module for CandidateFit-RAG.

Computes industry-standard Information Retrieval (IR) and QA metrics:
- Hit Rate @ K (Recall@K): Proportion of queries where at least one ground-truth chunk is retrieved in top-K.
- Mean Reciprocal Rank (MRR @ K): Average of reciprocal ranks (1 / rank) of the first relevant chunk.
- Precision @ K: Proportion of top-K retrieved chunks that are relevant.
- Refusal Accuracy: Proportion of unanswerable/out-of-corpus queries correctly caught by the safety guardrail.
"""

from typing import List, Dict, Any, Optional
import numpy as np

from src.chunking import Chunk
from src.generator import GenerationResponse


def is_chunk_relevant(
    chunk: Chunk,
    target_candidate: Optional[str],
    ground_truth_keywords: List[str]
) -> bool:
    """
    Evaluates whether a chunk satisfies relevance criteria for a labeled question.
    Requires matching candidate identity AND at least 50% of the ground-truth key facts.
    """
    if target_candidate is None:
        return False

    # Check candidate identity match
    chunk_candidate = chunk.candidate_name.lower()
    target_cand_clean = target_candidate.replace(".md", "").replace("_", " ").lower()

    name_matches = (
        chunk_candidate in target_cand_clean or
        target_cand_clean in chunk_candidate or
        any(part in chunk_candidate for part in target_cand_clean.split() if len(part) > 3)
    )

    if not name_matches:
        return False

    if not ground_truth_keywords:
        return True

    # Check keyword presence in chunk text
    text_lower = chunk.text.lower()
    matched_keywords = sum(1 for kw in ground_truth_keywords if kw.lower() in text_lower)
    
    # At least 1 keyword or half the keywords must be present
    threshold = max(1, len(ground_truth_keywords) // 2)
    return matched_keywords >= threshold


def compute_hit_rate_at_k(
    retrieved_chunks: List[Chunk],
    target_candidate: Optional[str],
    ground_truth_keywords: List[str],
    k: int
) -> float:
    """Returns 1.0 if a relevant chunk appears in the top-K, else 0.0."""
    top_k_chunks = retrieved_chunks[:k]
    for chunk in top_k_chunks:
        if is_chunk_relevant(chunk, target_candidate, ground_truth_keywords):
            return 1.0
    return 0.0


def compute_reciprocal_rank(
    retrieved_chunks: List[Chunk],
    target_candidate: Optional[str],
    ground_truth_keywords: List[str],
    max_k: int = 10
) -> float:
    """Computes Reciprocal Rank (1/rank) for the first relevant chunk within top-K."""
    for rank, chunk in enumerate(retrieved_chunks[:max_k], start=1):
        if is_chunk_relevant(chunk, target_candidate, ground_truth_keywords):
            return 1.0 / rank
    return 0.0


def compute_precision_at_k(
    retrieved_chunks: List[Chunk],
    target_candidate: Optional[str],
    ground_truth_keywords: List[str],
    k: int
) -> float:
    """Computes Precision@K: relevant chunks in top-K / K."""
    if k == 0:
        return 0.0
    top_k_chunks = retrieved_chunks[:k]
    relevant_count = sum(
        1 for chunk in top_k_chunks
        if is_chunk_relevant(chunk, target_candidate, ground_truth_keywords)
    )
    return relevant_count / k


def evaluate_refusal(is_answerable: bool, response: GenerationResponse) -> bool:
    """
    Evaluates whether the system appropriately handled answerability:
    - If question is unanswerable: Response MUST be a refusal (is_refusal == True).
    - If question is answerable: Response SHOULD provide an answer (is_refusal == False).
    """
    if not is_answerable:
        return response.is_refusal
    else:
        return not response.is_refusal
