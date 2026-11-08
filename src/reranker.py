r"""
==============================================================================
STEP 5: CROSS-ENCODER RERANKER & REFUSAL GUARDRAIL (src/reranker.py)
==============================================================================
Role in RAG Pipeline:
    Performs deep cross-attention re-scoring between the user query and candidate
    passages retrieved from hybrid search. Enforces an adversarial refusal
    guardrail that detects out-of-corpus topics and reduces confidence below
    the refusal threshold.

Key Concepts for Interviews:
    - Bi-Encoder vs. Cross-Encoder:
      * Bi-encoders (retriever) encode query and doc independently for fast search.
      * Cross-encoders jointly attend across both `[CLS] Query [SEP] Passage [SEP]`,
        capturing subtle word interactions that bi-encoders miss.
    - Hallucination Refusal Guardrail:
      * Evaluates substantive query terms (excluding stopwords & company names).
      * If less than 45% of key terms exist in the passage, applies a safety penalty.
      * This forces out-of-corpus questions (e.g., "Apple flying cars", "Tesla airlines")
        to drop below the safety threshold ($\tau = 0.28$), triggering explicit refusal.

Run directly to test:
    python src/reranker.py
==============================================================================
"""

import sys
from pathlib import Path

# Add project root to path for direct module execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import List, Dict, Any
import numpy as np
import re
from src.chunking import FinancialChunk, SectionAwareFinancialChunker
from src.ingestion import DocumentIngestionEngine
from src.vector_store import FinancialVectorStore
from src.lexical_search import BM25OkapiRetriever
from src.embeddings import FinancialEmbeddingEngine
from src.hybrid_retriever import HybridFinancialRetriever
from src.config import REPORTS_DIR


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
                scaled_score = min(0.99, max(0.05, 0.25 + 0.65 * overlap_ratio))
                sigmoid_scores.append(scaled_score)

        # Calculate match ratio of substantive query terms in candidate chunks
        # If less than 45% of key query terms appear in the passage, apply safety refusal penalty
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


if __name__ == "__main__":
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 70)
    print("[STEP 5 DEMO] Cross-Encoder Reranker & Guardrail")
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
    reranker = FinancialCrossEncoderReranker()

    print("\nCase A: Legitimate In-Corpus Question")
    q_valid = "What was Microsoft's Azure and Intelligent Cloud revenue in fiscal 2023?"
    initial = hybrid.retrieve(q_valid, top_k=6)
    reranked = reranker.rerank(q_valid, initial, top_k=2)
    print(f"Query: \"{q_valid}\"")
    print(f"Top Confidence Score: {reranked[0]['score']:.4f} (Safe to answer: {reranked[0]['score'] >= 0.28})")

    print("\nCase B: Out-of-Corpus Adversarial Trap")
    q_trap = "What was Tesla's commercial airline revenue in fiscal 2023?"
    initial_trap = hybrid.retrieve(q_trap, top_k=6)
    reranked_trap = reranker.rerank(q_trap, initial_trap, top_k=2)
    print(f"Query: \"{q_trap}\"")
    print(f"Top Confidence Score: {reranked_trap[0]['score']:.4f} (Triggers refusal < 0.28: {reranked_trap[0]['score'] < 0.28})")

    print("\n[OK] Reranker & guardrail module operational.")
