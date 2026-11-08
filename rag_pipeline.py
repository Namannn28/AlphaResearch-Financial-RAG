"""
CandidateFit-RAG: Simplified Core Pipeline.

This single file contains the entire RAG pipeline in plain, approachable Python.
Designed so that any fresher or early-career AI engineer can read it top-to-bottom
in 5 minutes and explain every single line in a technical interview with 100% confidence.

How it works (4 Simple Steps):
1. Load & Clean: Reads resumes from the data/resumes folder.
2. Section Chunking: Slices each resume by headings (## Work Experience, ## Skills) so
   a candidate's accomplishment metrics stay attached to the tools they used.
3. Hybrid Search: Combines exact keyword matching (BM25) with semantic meaning (MiniLM).
4. Guardrail: If top search similarity is low (< 0.32), refuses to answer instead of hallucinating.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import math
import re
import time
import numpy as np


# =====================================================================
# STEP 1: RESUME DATA STRUCTURES & INGESTION
# =====================================================================

@dataclass
class ResumeChunk:
    """Represents a clean, indexed section from a candidate's resume."""
    chunk_id: str
    candidate_name: str
    section_title: str
    text: str

    @property
    def citation(self) -> str:
        """Formatted citation tag for human-audited answers."""
        return f"[{self.candidate_name} | {self.section_title}]"


def clean_text(text: str) -> str:
    """Normalizes whitespace and removes weird line breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse 3+ newlines into 2 so paragraphs stay clean
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_resume_by_sections(content: str, filename: str) -> List[ResumeChunk]:
    """
    Splits a resume by markdown headers (## Experience, ## Skills).
    Why this matters in interviews:
    Standard RAG cuts every 200 words arbitrarily, which often cuts a bullet point in half.
    By splitting on section headers, an entire job role or skill block stays intact!
    """
    content = clean_text(content)
    
    # Extract candidate name from first # Heading
    name_match = re.search(r"^#\s+([A-Za-z\s\.\-]+)", content, re.MULTILINE)
    candidate_name = name_match.group(1).strip() if name_match else Path(filename).stem.replace("_", " ").title()

    # Split on markdown H2 headers (## Header)
    parts = re.split(r"(^##\s+[^\n]+)", content, flags=re.MULTILINE)
    
    chunks: List[ResumeChunk] = []
    current_header = "Summary"
    chunk_idx = 0

    i = 0
    while i < len(parts):
        part = parts[i].strip()
        if not part:
            i += 1
            continue

        if part.startswith("## "):
            current_header = part.replace("## ", "").strip()
            i += 1
            body = parts[i].strip() if i < len(parts) else ""
        else:
            body = part

        if len(body) >= 20:  # Only index meaningful sections
            chunk_text = f"Candidate: {candidate_name}\nSection: {current_header}\n\n{body}"
            chunks.append(ResumeChunk(
                chunk_id=f"{Path(filename).stem}_{chunk_idx:02d}",
                candidate_name=candidate_name,
                section_title=current_header,
                text=chunk_text
            ))
            chunk_idx += 1
        i += 1

    return chunks


# =====================================================================
# STEP 2: HYBRID SEARCH (BM25 KEYWORDS + DENSE SEMANTIC VECTORS)
# =====================================================================

class SimpleBM25Searcher:
    """
    BM25 Keyword Search built from scratch in 35 lines.
    Why we use this:
    If a recruiter searches for 'vLLM', 'ArgoCD', or 'OSCP', semantic vector embeddings
    can treat them as generic software terms. BM25 guarantees an exact hit!
    """
    def __init__(self, chunks: List[ResumeChunk]):
        self.chunks = chunks
        self.doc_tokens = [[w.lower() for w in re.findall(r"\w+", c.text)] for c in chunks]
        self.doc_lens = [len(tokens) for tokens in self.doc_tokens]
        self.avg_len = sum(self.doc_lens) / max(len(self.doc_lens), 1)
        self.vocab = {}
        for tokens in self.doc_tokens:
            for word in set(tokens):
                self.vocab[word] = self.vocab.get(word, 0) + 1

    def search(self, query: str, top_k: int = 5) -> List[tuple]:
        query_words = [w.lower() for w in re.findall(r"\w+", query)]
        scores = []
        n_docs = len(self.chunks)

        for i, tokens in enumerate(self.doc_tokens):
            score = 0.0
            doc_len = self.doc_lens[i]
            for qw in query_words:
                if qw in self.vocab:
                    tf = tokens.count(qw)
                    # Standard BM25 inverse document frequency
                    idf = math.log((n_docs - self.vocab[qw] + 0.5) / (self.vocab[qw] + 0.5) + 1.0)
                    score += idf * (tf * 2.5) / (tf + 1.5 * (0.25 + 0.75 * (doc_len / self.avg_len)))
            scores.append((self.chunks[i], score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class SimpleDenseEmbedder:
    """
    Embeds text chunks using HuggingFace sentence-transformers (MiniLM-L6-v2, 384 dimensions).
    If offline or running without torch, uses TF-IDF + SVD as a graceful local fallback.
    """
    def __init__(self):
        self.model = None
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        except Exception:
            self.model = None

    def embed(self, texts: List[str]) -> np.ndarray:
        if self.model is not None:
            vectors = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            return np.array(vectors, dtype=np.float32)
        else:
            # Simple TF-IDF fallback for 100% offline predictability
            from sklearn.feature_extraction.text import TfidfVectorizer
            vec = TfidfVectorizer(max_features=128)
            matrix = vec.fit_transform(texts).toarray()
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return (matrix / norms).astype(np.float32)


# =====================================================================
# STEP 3: END-TO-END RAG PIPELINE WITH SAFETY REFUSAL GUARDRAIL
# =====================================================================

class SimpleCandidateRAG:
    """
    The orchestrator: Ties together chunking, hybrid retrieval, and safety checks.
    """
    def __init__(self, resumes_dir: Path, refusal_threshold: float = 0.32):
        self.resumes_dir = resumes_dir
        self.refusal_threshold = refusal_threshold
        self.chunks: List[ResumeChunk] = []
        self.chunk_embeddings: Optional[np.ndarray] = None
        self.bm25: Optional[SimpleBM25Searcher] = None
        self.embedder = SimpleDenseEmbedder()
        self._build_index()

    def _build_index(self):
        """Loads all markdown resumes and prepares search indices."""
        for file_path in sorted(self.resumes_dir.glob("*.md")):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.chunks.extend(chunk_resume_by_sections(content, file_path.name))

        # Build BM25 index
        self.bm25 = SimpleBM25Searcher(self.chunks)

        # Build Dense vector index
        texts = [c.text for c in self.chunks]
        self.chunk_embeddings = self.embedder.embed(texts)

    def query(self, question: str, candidate_name_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Executes a grounded query with confidence guardrails.
        """
        start_time = time.perf_counter()

        # 1. Dense Semantic Vector Search (Cosine Similarity)
        q_emb = self.embedder.embed([question])[0]
        # Dot product with unit-normalized vectors equals cosine similarity
        dense_scores = np.dot(self.chunk_embeddings, q_emb)

        # 2. BM25 Keyword Search
        bm25_hits = {chunk.chunk_id: score for chunk, score in self.bm25.search(question, top_k=10)}

        # 3. Reciprocal Rank Fusion & Substantive Entity Check
        stopwords = {"what", "which", "who", "where", "how", "does", "did", "is", "was", "the",
                     "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with", "candidate",
                     "experience", "years", "has", "have", "had", "from", "by", "any", "about"}
        query_tokens = [w.lower() for w in re.findall(r"\w+", question) if w.lower() not in stopwords]

        # Collect candidate names and numbers to focus strictly on technical skills and tools
        all_name_words = {w.lower() for c in self.chunks for w in re.findall(r"\w+", c.candidate_name)}
        skill_query_tokens = [w for w in query_tokens if w not in all_name_words and not w.isdigit() and len(w) > 2]

        candidate_scores = []
        for i, chunk in enumerate(self.chunks):
            # Optional filter by candidate
            if candidate_name_filter and candidate_name_filter.lower() not in chunk.candidate_name.lower():
                continue

            d_score = float(dense_scores[i])
            b_score = bm25_hits.get(chunk.chunk_id, 0.0)

            # Check if at least one substantive skill keyword appears in the chunk text
            chunk_lower = chunk.text.lower()
            keyword_overlap = sum(1 for w in skill_query_tokens if w in chunk_lower)
            
            # If substantive skill terms were searched but none exist in the chunk,
            # clamp score to trigger the safety refusal guardrail
            if skill_query_tokens and keyword_overlap == 0:
                combined_score = 0.10
            else:
                combined_score = 0.5 * d_score + 0.5 * min(b_score / 4.0, 1.0)

            candidate_scores.append((chunk, combined_score, d_score))

        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        if not candidate_scores:
            return {
                "answer": "INSUFFICIENT_EVIDENCE: No matching records found.",
                "is_refusal": True,
                "confidence": 0.0,
                "latency_ms": latency_ms,
                "citations": []
            }

        top_chunk, top_score, top_dense = candidate_scores[0]

        # 4. Strict Safety Guardrail: Refuse to answer if confidence is too low
        # This prevents the system from hallucinating skills the candidate doesn't have!
        if top_score < self.refusal_threshold:
            return {
                "answer": (
                    f"INSUFFICIENT_EVIDENCE: The candidate resume corpus does not contain documented "
                    f"evidence to answer '{question}'. (Confidence {top_score:.2f} is below the safety "
                    f"threshold of {self.refusal_threshold:.2f}). Refusing to speculate to avoid hallucination."
                ),
                "is_refusal": True,
                "confidence": round(float(top_score), 3),
                "latency_ms": round(latency_ms, 1),
                "citations": []
            }

        # Format Grounded Answer
        clean_lines = [l.strip("-*• ") for l in top_chunk.text.split("\n") if l.strip() and not l.startswith("Candidate:") and not l.startswith("Section:")]
        evidence_snippet = clean_lines[0] if clean_lines else top_chunk.text[:200]

        answer = (
            f"Based on verified records for **{top_chunk.candidate_name}** ({top_chunk.citation}):\n\n"
            f"> \"{evidence_snippet}\"\n\n"
            f"*(Confidence: {top_score:.2f} | Retrieval Latency: {latency_ms:.1f}ms)*"
        )

        citations = [
            {"candidate": chunk.candidate_name, "section": chunk.section_title, "score": f"{score:.3f}"}
            for chunk, score, _ in candidate_scores[:3]
        ]

        return {
            "answer": answer,
            "is_refusal": False,
            "confidence": round(float(top_score), 3),
            "latency_ms": round(latency_ms, 1),
            "citations": citations
        }


# =====================================================================
# QUICK LOCAL TEST RUNNER
# =====================================================================
if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    resumes_path = current_dir / "data" / "resumes"

    print("--- Initializing CandidateFit-RAG (Simplified) ---")
    rag = SimpleCandidateRAG(resumes_dir=resumes_path)
    print(f"Indexed {len(rag.chunks)} chunks across all candidate resumes.\n")

    # Positive Query
    q1 = "Which candidate has experience with vLLM and TensorRT-LLM?"
    print(f"Q: {q1}")
    res1 = rag.query(q1)
    print(f"A: {res1['answer']}\n")

    # Negative Out-of-Corpus Query (Testing Safety Guardrail)
    q2 = "Does Marcus Vance have 8 years of COBOL experience on IBM mainframes?"
    print(f"Q: {q2}")
    res2 = rag.query(q2)
    print(f"A: {res2['answer']}\n")
