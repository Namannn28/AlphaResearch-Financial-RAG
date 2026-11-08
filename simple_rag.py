"""
==============================================================================
simple_rag.py - Standalone Financial Research Assistant (Interview-Ready RAG)
==============================================================================
Designed for freshers to read, run, and confidently explain during technical
AI/ML Engineer and GenAI interviews in under 5 minutes.

Core Steps Covered:
1. Ingestion: Reads SEC 10-K company annual reports (Apple, Microsoft, Tesla, etc.)
2. Chunking: Section-aware chunking preserving financial tables and headers.
3. Embeddings: In-memory dense embeddings (MiniLM) for semantic search.
4. BM25 Lexical Search: Keyword scoring for exact numbers ($383B, 18.2%) and tickers.
5. Hybrid Fusion (RRF): Combines vector and keyword search.
6. Guardrail Refusal: Refuses out-of-corpus queries when confidence < threshold.
7. LLM Answer Synthesis: Mistral API integration with automatic offline fallback.
==============================================================================
"""

import os
import re
import math
import time
import json
import urllib.request
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# ----------------------------------------------------------------------------
# 1. Data Models
# ----------------------------------------------------------------------------
@dataclass
class FinancialChunk:
    chunk_id: str
    company: str
    ticker: str
    section_title: str
    text: str

# ----------------------------------------------------------------------------
# 2. Ingestion & Section-Aware Chunking
# ----------------------------------------------------------------------------
def load_and_chunk_reports(reports_dir: Path) -> List[FinancialChunk]:
    """
    Reads markdown 10-K reports and chunks them by section headings (## Item X).
    Unlike naive token chunkers, this preserves financial statements and headers.
    """
    chunks: List[FinancialChunk] = []

    for file_path in reports_dir.glob("*.md"):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract basic header metadata
        company = file_path.stem.split("_")[0].capitalize()
        ticker = "N/A"
        for line in content.split("\n")[:15]:
            if "Ticker:" in line:
                ticker = line.split(":", 1)[1].strip()

        # Split on markdown headings (## or ###)
        sections = re.split(r"(^#{2,3}\s+[^\n]+)", content, flags=re.MULTILINE)
        current_title = "Overview"

        for token in sections:
            token = token.strip()
            if not token:
                continue
            if token.startswith("## ") or token.startswith("### "):
                current_title = token.lstrip("#").strip()
            else:
                chunk_id = f"{file_path.stem}_{len(chunks)}"
                full_text = f"[{company} ({ticker}) | {current_title}]\n{token}"
                chunks.append(FinancialChunk(
                    chunk_id=chunk_id,
                    company=company,
                    ticker=ticker,
                    section_title=current_title,
                    text=full_text
                ))

    return chunks

# ----------------------------------------------------------------------------
# 3. BM25 Lexical Search (Keyword Matching)
# ----------------------------------------------------------------------------
class SimpleBM25:
    """Exact keyword search for currency figures ($ billions), percentages, and tickers."""

    def __init__(self, chunks: List[FinancialChunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_lens = [len(self.tokenize(c.text)) for c in chunks]
        self.avg_doc_len = sum(self.doc_lens) / max(1, len(chunks))

        # Build term frequencies and document frequencies
        self.term_freqs = []
        self.doc_freqs: Dict[str, int] = {}
        for c in chunks:
            tokens = self.tokenize(c.text)
            tf = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self.term_freqs.append(tf)
            for t in tf.keys():
                self.doc_freqs[t] = self.doc_freqs.get(t, 0) + 1

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return [t.lower() for t in re.findall(r"[\$]?[a-zA-Z0-9_\.]+%?", text) if len(t) > 1]

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        tokens = self.tokenize(query)
        scores = [0.0] * len(self.chunks)
        n_docs = len(self.chunks)

        for t in tokens:
            if t not in self.doc_freqs:
                continue
            df = self.doc_freqs[t]
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

            for idx in range(n_docs):
                tf = self.term_freqs[idx].get(t, 0)
                if tf > 0:
                    numerator = tf * (self.k1 + 1.0)
                    denominator = tf + self.k1 * (1.0 - self.b + self.b * (self.doc_lens[idx] / self.avg_doc_len))
                    scores[idx] += idf * (numerator / denominator)

        ranked = sorted(range(n_docs), key=lambda i: scores[i], reverse=True)
        return [{"chunk": self.chunks[i], "score": scores[i], "rank": r + 1} for r, i in enumerate(ranked[:top_k]) if scores[i] > 0]

# ----------------------------------------------------------------------------
# 4. Dense Vector Embeddings (MiniLM / Fallback)
# ----------------------------------------------------------------------------
class SimpleEmbedder:
    """Computes dense normalized embeddings using SentenceTransformers."""

    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        except Exception:
            self.model = None

    def embed(self, texts: List[str]):
        import numpy as np
        if self.model is not None:
            return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        # Fast fallback if sentence-transformers is offline
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf = TfidfVectorizer(max_features=384, stop_words="english").fit_transform(texts).toarray()
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        return (tfidf / norms).astype(np.float32)

# ----------------------------------------------------------------------------
# 5. Hybrid Search & Guardrail Answer Synthesis (LangChain MistralAI)
# ----------------------------------------------------------------------------
class FinancialRAG:
    """Coordinates retrieval, confidence evaluation, and answer generation using LangChain."""

    def __init__(self, reports_dir: Path, mistral_api_key: Optional[str] = None):
        import numpy as np
        self.np = np
        self.chunks = load_and_chunk_reports(reports_dir)
        self.bm25 = SimpleBM25(self.chunks)
        self.embedder = SimpleEmbedder()
        self.chunk_vectors = self.embedder.embed([c.text for c in self.chunks])
        self.mistral_key = mistral_api_key or os.getenv("MISTRAL_API_KEY", "")

        # Initialize LangChain ChatMistralAI model
        self.chat_model = None
        if self.mistral_key:
            try:
                from langchain_mistralai import ChatMistralAI
                self.chat_model = ChatMistralAI(
                    model="mistral-small-latest",
                    mistral_api_key=self.mistral_key,
                    temperature=0.1
                )
            except Exception:
                self.chat_model = None

    def answer_query(self, query: str, confidence_threshold: float = 0.28) -> str:
        """Runs hybrid search, evaluates confidence, and produces an answer with citations."""
        start = time.perf_counter()

        # Step 1: BM25 Lexical hits
        bm25_hits = self.bm25.search(query, top_k=6)

        # Step 2: Dense Semantic hits
        q_vec = self.embedder.embed([query])[0]
        dense_scores = self.np.dot(self.chunk_vectors, q_vec)
        dense_top_indices = self.np.argsort(dense_scores)[::-1][:6]

        # Step 3: Reciprocal Rank Fusion (RRF)
        fused_scores: Dict[str, float] = {}
        for rank, hit in enumerate(bm25_hits, start=1):
            cid = hit["chunk"].chunk_id
            fused_scores[cid] = fused_scores.get(cid, 0.0) + (1.0 / (60 + rank))

        for rank, idx in enumerate(dense_top_indices, start=1):
            cid = self.chunks[idx].chunk_id
            fused_scores[cid] = fused_scores.get(cid, 0.0) + (1.0 / (60 + rank))

        # Sort top fused chunk
        best_cid = max(fused_scores.items(), key=lambda x: x[1])[0]
        best_chunk = next(c for c in self.chunks if c.chunk_id == best_cid)
        top_idx = next(i for i, c in enumerate(self.chunks) if c.chunk_id == best_cid)
        confidence = float(dense_scores[top_idx])

        # Calculate match ratio of substantive query terms in retrieved chunk
        # If less than 45% of key query terms appear in the chunk, heavily penalize to prevent hallucination
        stopwords = {
            "what", "was", "did", "the", "in", "and", "for", "with", "from", "how", "much",
            "apple", "tesla", "microsoft", "amazon", "alphabet", "google", "inc", "corp",
            "total", "fiscal", "year", "2023", "2022", "annual", "report", "company"
        }
        raw_words = re.findall(r"[a-zA-Z0-9]+", query.lower())
        substantive_terms = [w for w in raw_words if len(w) > 2 and w not in stopwords]

        if substantive_terms:
            matched_terms = [w for w in substantive_terms if w in best_chunk.text.lower()]
            match_ratio = len(matched_terms) / len(substantive_terms)
            if match_ratio < 0.45:
                confidence *= 0.20  # Guardrail penalty for out-of-corpus queries

        latency_ms = (time.perf_counter() - start) * 1000.0

        # Step 4: Safety Guardrail (Refusal on low confidence or out-of-corpus query)
        if confidence < confidence_threshold:
            return (
                f"[REFUSAL - INSUFFICIENT_EVIDENCE]: The 10-K filings do not contain documented evidence to answer "
                f"'{query}'. (Confidence {confidence:.2f} < threshold {confidence_threshold}). "
                f"Refusing to speculate to prevent financial hallucination."
            )

        # Step 5: Answer with LangChain ChatMistralAI (or clean offline extractive fallback)
        if self.chat_model is not None:
            try:
                from langchain_core.messages import SystemMessage, HumanMessage
                messages = [
                    SystemMessage(content="You are a professional financial analyst. Answer the user question using ONLY the provided verified 10-K context, with citations."),
                    HumanMessage(content=f"Context:\n{best_chunk.text}\n\nQuestion: {query}")
                ]
                response = self.chat_model.invoke(messages)
                answer = response.content.strip()
                return f"{answer}\n\n*(Citations: [{best_chunk.company} ({best_chunk.ticker}) | {best_chunk.section_title}] | Provider: LangChain ChatMistralAI | Latency: {latency_ms:.1f}ms)*"
            except Exception:
                pass  # Fallback to offline extraction

        elif self.mistral_key:
            try:
                answer = self._call_mistral(query, best_chunk.text)
                return f"{answer}\n\n*(Citations: [{best_chunk.company} ({best_chunk.ticker}) | {best_chunk.section_title}] | Provider: Mistral API | Latency: {latency_ms:.1f}ms)*"
            except Exception:
                pass  # Fallback to offline extraction

        # Offline extractive summary
        return (
            f"Based on verified 10-K filings for **{best_chunk.company} ({best_chunk.ticker})** "
            f"([*{best_chunk.section_title}*]):\n\n"
            f"> {best_chunk.text.splitlines()[-1] if len(best_chunk.text.splitlines()) > 1 else best_chunk.text}\n\n"
            f"*(Confidence: {confidence:.2f} | Latency: {latency_ms:.1f}ms)*"
        )

    def _call_mistral(self, query: str, context: str) -> str:
        url = "https://api.mistral.ai/v1/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.mistral_key}"}
        payload = {
            "model": "mistral-small-latest",
            "messages": [
                {"role": "system", "content": "You are a financial analyst. Answer using ONLY the provided context with citations."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ],
            "temperature": 0.1,
            "max_tokens": 300
        }
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))["choices"][0]["message"]["content"].strip()

# ----------------------------------------------------------------------------
# 6. Demonstration Entry Point
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    reports_path = Path(__file__).parent / "data" / "financial_reports"
    print("--- Initializing Financial Research Assistant RAG ---")
    rag = FinancialRAG(reports_path)
    print(f"Indexed {len(rag.chunks)} chunks across Apple, Microsoft, Tesla, Alphabet, Amazon.\n")

    # Positive Query (Factual number extraction)
    q1 = "What was Apple's total net sales and iPhone revenue in fiscal 2023?"
    print(f"Q: {q1}")
    print(f"A: {rag.answer_query(q1)}\n")

    # Negative Adversarial Query (Out of corpus refusal guardrail)
    q2 = "What was Tesla's commercial airline revenue in fiscal 2023?"
    print(f"Q: {q2}")
    print(f"A: {rag.answer_query(q2)}\n")
