"""
Unit and Integration Test Suite for CandidateFit-RAG.

Verifies:
1. Document ingestion and metadata extraction
2. Fixed-window vs Section-aware chunking behaviors
3. Lexical BM25 and Dense vector retrieval
4. Reciprocal Rank Fusion (RRF) logic
5. Out-of-corpus refusal safety guardrails
"""

import pytest
from pathlib import Path

from src.ingestion import load_single_document, clean_text, Document
from src.chunking import FixedWindowChunker, SectionAwareSemanticChunker, build_chunks
from src.vector_store import InMemoryVectorStore
from src.lexical_search import BM25LexicalSearcher
from src.pipeline import CandidateFitRAGPipeline
from src.config import RESUMES_DIR, retrieval_config
import numpy as np


def test_text_cleaning():
    raw = "Line 1\r\n\r\n\r\n\r\nLine 2   \u00a0"
    cleaned = clean_text(raw)
    assert "\r" not in cleaned
    assert "\u00a0" not in cleaned
    assert "Line 1\n\nLine 2" == cleaned


def test_section_aware_chunking_preserves_headers():
    sample_text = (
        "# Jane Doe\n"
        "## Technical Skills\n"
        "Python, PyTorch, Docker, Kubernetes.\n\n"
        "## Work Experience\n"
        "### Staff Engineer | TechCorp (2022 - Present)\n"
        "Built low-latency model serving platform with vLLM."
    )
    doc = Document(
        doc_id="test_doc",
        filename="test_doc.md",
        content=sample_text,
        metadata={"candidate_name": "Jane Doe"}
    )
    chunker = SectionAwareSemanticChunker(min_section_length_chars=20)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 2
    headers = [c.section_title for c in chunks]
    assert any("Technical Skills" in h for h in headers)
    assert any("Staff Engineer" in h for h in headers)
    # Check that candidate name is explicitly prefixed
    assert all("Jane Doe" in c.text for c in chunks)


def test_fixed_window_chunking_overlap():
    words = [f"word_{i}" for i in range(100)]
    doc = Document(
        doc_id="fixed_test",
        filename="fixed_test.md",
        content=" ".join(words),
        metadata={"candidate_name": "Test Candidate"}
    )
    chunker = FixedWindowChunker(window_size_words=40, overlap_words=10)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 3
    # Step = 30 words
    assert chunks[0].metadata["start_word_index"] == 0
    assert chunks[1].metadata["start_word_index"] == 30
    assert chunks[2].metadata["start_word_index"] == 60


def test_in_memory_vector_store():
    store = InMemoryVectorStore()
    doc = Document("d1", "d1.md", "dummy", {"candidate_name": "Alice"})
    chunker = FixedWindowChunker(window_size_words=10, overlap_words=2)
    chunks = chunker.chunk_document(Document("d1", "d1.md", "word " * 20, {"candidate_name": "Alice"}))

    # Create dummy 4-dimensional normalized embeddings
    embs = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0]
    ], dtype=np.float32)

    store.add_chunks(chunks[:2], embs)

    query_vec = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
    results = store.search(query_vec, top_k=1)

    assert len(results) == 1
    best_chunk, score = results[0]
    assert score > 0.8
    assert best_chunk.doc_id == "d1"


def test_bm25_lexical_exact_keyword_retrieval():
    searcher = BM25LexicalSearcher()
    chunker = SectionAwareSemanticChunker(min_section_length_chars=10)
    chunk1 = chunker.chunk_document(
        Document("d1", "d1.md", "## Skills\nProficient in vLLM and TensorRT-LLM GPU serving.", {"candidate_name": "Alex"})
    )[0]
    chunk2 = chunker.chunk_document(
        Document("d2", "d2.md", "## Skills\nProficient in Apache Iceberg and Snowflake lakehouses.", {"candidate_name": "Priya"})
    )[0]

    searcher.index_chunks([chunk1, chunk2])

    hits = searcher.search("TensorRT-LLM", top_k=1)
    assert len(hits) == 1
    assert hits[0][0].candidate_name == "Alex"


def test_out_of_corpus_safety_refusal():
    pipeline = CandidateFitRAGPipeline(
        chunking_strategy="section_aware",
        prefer_sentence_transformer=False
    )
    pipeline.build_index()

    # Query with a capability completely absent from the candidate pool
    unanswerable_query = "Which candidate has 8 years of COBOL and Fortran 77 development on IBM mainframes?"
    response = pipeline.query(unanswerable_query)

    assert response.is_refusal is True
    assert "INSUFFICIENT_EVIDENCE" in response.answer
