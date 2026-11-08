"""
Lexical Keyword Search Module for CandidateFit-RAG.

Technical resumes contain crucial exact-match entities (e.g., 'vLLM', 'ArgoCD',
'Delta Lake', 'Next.js 14', 'OSCP', 'SOC 2'). Pure dense retrieval can miss
these exact acronyms due to semantic smoothing. This module provides BM25 / TF-IDF
lexical retrieval to pair with dense semantic search.
"""

from typing import List, Tuple, Optional
import math
import re

from src.chunking import Chunk


def tokenize_for_lexical_search(text: str) -> List[str]:
    """
    Tokenizes text while preserving technical symbols like dots, hyphens, and pluses
    (e.g., 'c++', 'next.js', 'gpt-4').
    """
    clean = text.lower()
    # Replace markdown punctuation but preserve identifiers
    tokens = re.findall(r"[a-z0-9][a-z0-9\.\-\+]*", clean)
    return tokens


class BM25LexicalSearcher:
    """
    BM25Okapi ranking implementation built from first principles.
    Guarantees zero external dependency failure and gives full transparency
    over IDF calculation and length normalization.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: List[Chunk] = []
        self.corpus_size: int = 0
        self.avg_doc_len: float = 0.0
        self.doc_lengths: List[int] = []
        self.doc_token_counts: List[dict] = []
        self.doc_frequencies: dict = {}
        self.idf: dict = {}

    def index_chunks(self, chunks: List[Chunk]):
        """Indexes candidate chunks and computes Inverse Document Frequency (IDF)."""
        self.chunks = chunks
        self.corpus_size = len(chunks)
        self.doc_lengths = []
        self.doc_token_counts = []
        self.doc_frequencies = {}

        if self.corpus_size == 0:
            return

        total_length = 0
        for chunk in chunks:
            tokens = tokenize_for_lexical_search(chunk.text)
            length = len(tokens)
            self.doc_lengths.append(length)
            total_length += length

            token_counts = {}
            unique_tokens = set()
            for token in tokens:
                token_counts[token] = token_counts.get(token, 0) + 1
                unique_tokens.add(token)

            self.doc_token_counts.append(token_counts)

            for token in unique_tokens:
                self.doc_frequencies[token] = self.doc_frequencies.get(token, 0) + 1

        self.avg_doc_len = total_length / self.corpus_size if self.corpus_size > 0 else 0.0

        # Calculate BM25 IDF: ln((N - df + 0.5) / (df + 0.5) + 1.0)
        self.idf = {}
        for token, df in self.doc_frequencies.items():
            idf_score = math.log(((self.corpus_size - df + 0.5) / (df + 0.5)) + 1.0)
            self.idf[token] = max(idf_score, 0.05)  # floor to prevent negative IDF for common terms

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_name_filter: Optional[str] = None
    ) -> List[Tuple[Chunk, float]]:
        """
        Calculates BM25 scores for the input query across all indexed chunks.
        """
        if self.corpus_size == 0:
            return []

        query_tokens = tokenize_for_lexical_search(query)
        if not query_tokens:
            return []

        scores: List[float] = [0.0] * self.corpus_size

        for i in range(self.corpus_size):
            chunk = self.chunks[i]
            
            # Filter check
            if candidate_name_filter and candidate_name_filter.lower() not in chunk.candidate_name.lower():
                continue

            doc_len = self.doc_lengths[i]
            token_counts = self.doc_token_counts[i]
            score = 0.0

            for token in query_tokens:
                if token not in token_counts:
                    continue

                term_frequency = token_counts[token]
                idf_val = self.idf.get(token, 0.0)

                # Standard BM25 term weighting formula
                numerator = term_frequency * (self.k1 + 1.0)
                denominator = term_frequency + self.k1 * (1.0 - self.b + self.b * (doc_len / (self.avg_doc_len or 1.0)))
                score += idf_val * (numerator / denominator)

            scores[i] = score

        # Rank documents by score descending
        ranked_indices = sorted(
            [i for i in range(self.corpus_size) if scores[i] > 0.0],
            key=lambda idx: scores[idx],
            reverse=True
        )[:top_k]

        results = [(self.chunks[idx], scores[idx]) for idx in ranked_indices]
        return results
