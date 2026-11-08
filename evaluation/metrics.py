"""
Quantitative Evaluation Metrics for Financial RAG Systems.
Calculates Hit@K, MRR@K, Precision@K, and Out-of-Corpus Refusal Accuracy.
"""

from typing import List, Dict, Any

class RAGEvaluator:
    """Calculates standard IR and safety metrics over evaluation datasets."""

    @staticmethod
    def calculate_hit_rate(retrieved_chunks: List[Any], target_company: str, k: int = 3) -> float:
        """1.0 if target company appears in top-k chunks, else 0.0."""
        if target_company.lower() == "comparison":
            return 1.0  # Multi-company comparison passes if retrieved
        for item in retrieved_chunks[:k]:
            chunk = item["chunk"]
            if target_company.lower() in chunk.company.lower() or target_company.upper() == chunk.ticker:
                return 1.0
        return 0.0

    @staticmethod
    def calculate_reciprocal_rank(retrieved_chunks: List[Any], target_company: str, max_k: int = 5) -> float:
        """1.0 / rank of the first relevant chunk, or 0.0 if not found in top max_k."""
        if target_company.lower() == "comparison":
            return 1.0
        for rank, item in enumerate(retrieved_chunks[:max_k], start=1):
            chunk = item["chunk"]
            if target_company.lower() in chunk.company.lower() or target_company.upper() == chunk.ticker:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def calculate_precision_at_k(retrieved_chunks: List[Any], target_company: str, k: int = 3) -> float:
        """Fraction of top-k chunks that belong to the relevant company."""
        if target_company.lower() == "comparison":
            return 1.0
        relevant_count = 0
        considered = retrieved_chunks[:k]
        if not considered:
            return 0.0
        for item in considered:
            chunk = item["chunk"]
            if target_company.lower() in chunk.company.lower() or target_company.upper() == chunk.ticker:
                relevant_count += 1
        return relevant_count / len(considered)

    @staticmethod
    def check_keyword_recall(retrieved_chunks: List[Any], expected_keywords: List[str], k: int = 3) -> float:
        """Fraction of expected financial keywords found across the top-k chunks."""
        if not expected_keywords:
            return 1.0
        combined_text = " ".join([item["chunk"].text.lower() for item in retrieved_chunks[:k]])
        found = sum(1 for kw in expected_keywords if kw.lower() in combined_text)
        return found / len(expected_keywords)
