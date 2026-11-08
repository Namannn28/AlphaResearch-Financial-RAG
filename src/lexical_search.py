"""
BM25 Lexical Keyword Search for Financial Data.
Handles exact matches for ticker symbols (AAPL, MSFT, TSLA, AMZN),
currency metrics ($383.29 billion), and percentages (44.1%).
"""

from typing import List, Dict, Any, Optional
import math
import re
from src.chunking import FinancialChunk

class BM25OkapiRetriever:
    """Standard Okapi BM25 implementation for exact keyword search."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: List[FinancialChunk] = []
        self.corpus_size = 0
        self.avg_doc_len = 0.0
        self.doc_lens: List[int] = []
        self.doc_freqs: Dict[str, int] = {}
        self.term_freqs: List[Dict[str, int]] = []

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenizes text preserving financial figures, percentages, and tickers."""
        # Keep words, currency amounts like $383.29, percentages like 44.1%, and numbers
        tokens = re.findall(r"[\$]?[a-zA-Z0-9_\.]+%?", text.lower())
        return [t for t in tokens if len(t) > 1 or t in {"$", "%"}]

    def index_chunks(self, chunks: List[FinancialChunk]):
        """Builds inverted index and term frequency tables over chunks."""
        self.chunks = chunks
        self.corpus_size = len(chunks)
        self.doc_lens = []
        self.doc_freqs = {}
        self.term_freqs = []

        total_length = 0
        for chunk in chunks:
            tokens = self.tokenize(chunk.text)
            self.doc_lens.append(len(tokens))
            total_length += len(tokens)

            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self.term_freqs.append(tf)

            for unique_t in tf.keys():
                self.doc_freqs[unique_t] = self.doc_freqs.get(unique_t, 0) + 1

        self.avg_doc_len = (total_length / self.corpus_size) if self.corpus_size > 0 else 1.0

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_company: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Calculates BM25 score for query across all indexed chunks."""
        query_tokens = self.tokenize(query)
        if not query_tokens or self.corpus_size == 0:
            return []

        scores: List[float] = [0.0] * self.corpus_size

        for token in query_tokens:
            if token not in self.doc_freqs:
                continue

            # Inverse Document Frequency (IDF) with BM25 smoothing
            df = self.doc_freqs[token]
            idf = math.log(1.0 + (self.corpus_size - df + 0.5) / (df + 0.5))

            for idx in range(self.corpus_size):
                tf = self.term_freqs[idx].get(token, 0)
                if tf == 0:
                    continue

                doc_len = self.doc_lens[idx]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                scores[idx] += idf * (numerator / denominator)

        # Rank and filter
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results: List[Dict[str, Any]] = []

        for idx in ranked_indices:
            if scores[idx] <= 0.0:
                break
            chunk = self.chunks[idx]
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
