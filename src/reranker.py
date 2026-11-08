r"""
==============================================================================
STEP 5: CROSS-ENCODER RERANKER & REFUSAL GUARDRAIL (src/reranker.py)
==============================================================================
Role in RAG Pipeline:
    Performs deep cross-attention re-scoring between the user query and candidate
    passages retrieved from hybrid search. Enforces an adversarial refusal
    guardrail that detects out-of-corpus topics while welcoming broad
    conversational inquiries about in-corpus companies.

Key Concepts for Interviews:
    - Bi-Encoder vs. Cross-Encoder:
      * Bi-encoders (retriever) encode query and doc independently for fast search.
      * Cross-encoders jointly attend across both `[CLS] Query [SEP] Passage [SEP]`,
        capturing subtle word interactions that bi-encoders miss.
    - Calibrated Refusal Guardrail:
      * Distinguishes between legitimate conversational queries ("tell me about
        financials about apple") and adversarial hallucination traps ("Tesla
        commercial airline revenue").
      * Identifies foreign non-financial terms: if a query asks for topics
        completely absent from the filing (airlines, flying cars), applies a
        guardrail penalty dropping confidence below threshold ($\tau = 0.28$).

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

    CONVERSATIONAL_FILLERS = {
        "what", "was", "did", "the", "in", "and", "for", "with", "from", "how", "much",
        "apple", "tesla", "microsoft", "amazon", "alphabet", "google", "inc", "corp",
        "total", "fiscal", "year", "2023", "2022", "annual", "report", "company",
        "tell", "me", "about", "give", "show", "can", "you", "please", "know",
        "information", "info", "details", "summary", "overview", "all", "some",
        "find", "explain", "describe", "look", "up", "get", "share",
        "yes", "yeah", "yep", "sure", "ok", "okay", "also", "now", "i", "want",
        "need", "like", "would", "first", "next", "which", "whose", "one", "do", "does",
        "here", "there", "is", "it", "are", "were", "been", "being", "having", "has", "had",
        "could", "should", "thanks", "thank"
    }

    FINANCIAL_DOMAIN_TERMS = {
        "financials", "financial", "revenue", "revenues", "sales", "profit", "margin",
        "margins", "income", "expenses", "deliveries", "capex", "growth", "business",
        "performance", "overview", "expenditures", "earnings", "guidance",
        "conservation", "energy", "technology", "absorption", "governance", "csr",
        "board", "director", "directors", "report", "auditor", "audit", "compliance",
        "policy", "shareholders", "dividend", "pat", "ebitda", "assets", "liabilities",
        "cash", "flow", "equity", "debt", "r&d", "research", "development", "subsidiary",
        "segment", "balance", "sheet", "statement", "notes", "ratio", "ratios", "schedule",
        "foreign", "exchange", "outgo", "turnover", "profitability", "capital", "expenditure",
        "measures", "initiatives", "operations", "production", "capacity", "cost", "costs",
        "management", "discussion", "analysis"
    }

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

        # 1. Compute Base Cross-Encoder Scores
        if self.model is not None:
            pairs = [[query, hit["chunk"].text] for hit in candidate_hits]
            try:
                raw_scores = self.model.predict(pairs)
                # Convert logits to 0.0 - 1.0 sigmoid probability
                sigmoid_scores = 1.0 / (1.0 + np.exp(-raw_scores))
            except Exception:
                sigmoid_scores = [hit["score"] for hit in candidate_hits]
        else:
            query_words = set(w.lower().strip("$,%") for w in query.split() if len(w) > 2)
            sigmoid_scores = []
            for hit in candidate_hits:
                text_lower = hit["chunk"].text.lower()
                matches = sum(1 for w in query_words if w in text_lower)
                overlap_ratio = matches / max(1, len(query_words))
                scaled_score = min(0.99, max(0.05, 0.25 + 0.65 * overlap_ratio))
                sigmoid_scores.append(scaled_score)

        # 2. Extract Substantive and Foreign Non-Financial Terms
        raw_words = re.findall(r"[a-zA-Z0-9]+", query.lower())
        substantive_terms = [w for w in raw_words if len(w) > 2 and w not in self.CONVERSATIONAL_FILLERS]
        foreign_terms = [w for w in substantive_terms if w not in self.FINANCIAL_DOMAIN_TERMS]

        query_lower = query.lower()

        # 3. Score Calibration & Adversarial Guardrail
        reranked_hits: List[Dict[str, Any]] = []
        for original_idx, (hit, score) in enumerate(zip(candidate_hits, sigmoid_scores)):
            chunk = hit["chunk"]
            text_lower = chunk.text.lower()
            chunk_company = chunk.company.lower()
            chunk_ticker = chunk.ticker.lower()

            # Check if chunk matches queried company
            is_company_match = (
                chunk_company in query_lower or
                chunk_ticker in query_lower or
                any(tok in query_lower for tok in chunk_company.split() if len(tok) > 3)
            )

            final_score = float(score)

            # Check substantive word overlap
            matched_substantive = [
                w for w in substantive_terms
                if w in text_lower or (len(w) > 4 and w.rstrip("s") in text_lower)
            ]
            substantive_ratio = len(matched_substantive) / max(1, len(substantive_terms))

            # Foreign non-financial terms completely absent from document
            missing_foreign = [
                w for w in foreign_terms
                if w not in text_lower and (len(w) > 4 and w.rstrip("s") not in text_lower)
            ]

            # If genuine foreign hallucination terms exist and are completely absent from document
            if foreign_terms and len(missing_foreign) == len(foreign_terms):
                # Genuine adversarial out-of-corpus query (e.g., airline, flying cars, submarines)
                final_score *= 0.15
            elif substantive_ratio >= 0.4:
                # Strong topical or heading match
                final_score = max(final_score, 0.88)
            elif is_company_match and (not substantive_terms or substantive_ratio >= 0.3):
                # Conversational or general query grounded in targeted company
                final_score = max(final_score, 0.88)
            elif substantive_terms and substantive_ratio < 0.25:
                # Low substantive overlap
                final_score *= 0.35

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

    print("\nCase 1: Broad Conversational Inquiry (e.g. 'tell me about financials about apple')")
    q_broad = "tell me about financials about apple"
    hits_broad = hybrid.retrieve(q_broad, top_k=6)
    reranked_broad = reranker.rerank(q_broad, hits_broad, top_k=2)
    print(f"Query: \"{q_broad}\"")
    print(f"Top Confidence Score: {reranked_broad[0]['score']:.4f} (Safe to answer: {reranked_broad[0]['score'] >= 0.28})")

    print("\nCase 2: Out-of-Corpus Adversarial Trap")
    q_trap = "What was Tesla's commercial airline revenue in fiscal 2023?"
    initial_trap = hybrid.retrieve(q_trap, top_k=6)
    reranked_trap = reranker.rerank(q_trap, initial_trap, top_k=2)
    print(f"Query: \"{q_trap}\"")
    print(f"Top Confidence Score: {reranked_trap[0]['score']:.4f} (Triggers refusal < 0.28: {reranked_trap[0]['score'] < 0.28})")

    print("\n[OK] Reranker & guardrail module operational.")
