"""
Cross-Encoder Reranker for Precision Financial Chunk Re-ordering.
Uses cross-encoder/ms-marco-MiniLM-L-6-v2 with fallback scoring.
"""

from typing import List, Dict, Any
import numpy as np

class FinancialCrossEncoderReranker:
    """
    Reranks candidate chunks retrieved from hybrid search using full cross-attention.
    Captures nuanced relationships between the financial query and candidate passages.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        enable_neural: bool = True
    ):
        self.model_name = model_name
        self.model = None

        if enable_neural:
            try:
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder(model_name)
            except Exception:
                self.model = None

    def rerank(
        self,
        query: str,
        candidate_hits: List[Dict[str, Any]],
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Reranks candidate hits and computes confidence scores."""
        if not candidate_hits:
            return []

        # If neural cross-encoder is loaded
        if self.model is not None:
            pairs = [[query, hit["chunk"].text] for hit in candidate_hits]
            try:
                raw_scores = self.model.predict(pairs)
                # Convert logits to 0.0 - 1.0 sigmoid probability
                sigmoid_scores = 1.0 / (1.0 + np.exp(-raw_scores))
            except Exception:
                sigmoid_scores = [hit["score"] for hit in candidate_hits]
        else:
            # High-fidelity lexical/substantive overlap fallback scorer
            query_words = set(w.lower().strip("$,%") for w in query.split() if len(w) > 2)
            sigmoid_scores = []
            for hit in candidate_hits:
                text_lower = hit["chunk"].text.lower()
                matches = sum(1 for w in query_words if w in text_lower)
                overlap_ratio = matches / max(1, len(query_words))
                # Combine initial retrieval score and overlap
                scaled_score = min(0.99, max(0.05, 0.25 + 0.65 * overlap_ratio))
                sigmoid_scores.append(scaled_score)

        # Calculate match ratio of substantive query terms in candidate chunks
        # If less than 45% of key query terms appear in the passage, apply safety refusal penalty
        import re
        stopwords = {
            "what", "was", "did", "the", "in", "and", "for", "with", "from", "how", "much",
            "apple", "tesla", "microsoft", "amazon", "alphabet", "google", "inc", "corp",
            "total", "fiscal", "year", "2023", "2022", "annual", "report", "company"
        }
        raw_words = re.findall(r"[a-zA-Z0-9]+", query.lower())
        substantive_terms = [w for w in raw_words if len(w) > 2 and w not in stopwords]

        # Attach rerank score and trace rank lift
        reranked_hits: List[Dict[str, Any]] = []
        for original_idx, (hit, score) in enumerate(zip(candidate_hits, sigmoid_scores)):
            final_score = float(score)
            if substantive_terms:
                text_lower = hit["chunk"].text.lower()
                matched_terms = [w for w in substantive_terms if w in text_lower]
                match_ratio = len(matched_terms) / len(substantive_terms)
                if match_ratio < 0.45:
                    final_score *= 0.20  # Guardrail penalty for out-of-corpus queries

            reranked_hits.append({
                "chunk": hit["chunk"],
                "score": final_score,
                "original_rank": original_idx + 1,
                "dense_rank": hit.get("dense_rank"),
                "bm25_rank": hit.get("bm25_rank")
            })

        # Sort by rerank score descending
        reranked_hits.sort(key=lambda x: x["score"], reverse=True)

        for new_rank, item in enumerate(reranked_hits[:top_k], start=1):
            item["new_rank"] = new_rank
            item["rank_lift"] = item["original_rank"] - new_rank

        return reranked_hits[:top_k]
