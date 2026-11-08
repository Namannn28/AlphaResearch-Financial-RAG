"""
Unit and Integration Tests for Financial Research Assistant RAG Pipeline.
Verifies ingestion, section-aware chunking, BM25 exact matching,
vector cosine search, hybrid fusion, and refusal guardrails.
"""

import pytest
from pathlib import Path
from src.config import config, REPORTS_DIR
from src.ingestion import DocumentIngestionEngine
from src.chunking import SectionAwareFinancialChunker, FixedWindowChunker
from src.embeddings import FinancialEmbeddingEngine
from src.vector_store import FinancialVectorStore
from src.lexical_search import BM25OkapiRetriever
from src.hybrid_retriever import HybridFinancialRetriever
from src.pipeline import FinancialRAGPipeline

@pytest.fixture
def sample_reports():
    engine = DocumentIngestionEngine()
    docs = engine.load_all_reports(REPORTS_DIR)
    assert len(docs) >= 5, "Expected at least 5 financial reports (Apple, Microsoft, Tesla, Alphabet, Amazon)"
    return docs

def test_document_ingestion(sample_reports):
    """Verifies that all 10-K files are loaded and metadata parsed properly."""
    tickers = {doc.ticker for doc in sample_reports}
    assert "AAPL" in tickers
    assert "MSFT" in tickers
    assert "TSLA" in tickers
    assert "GOOGL" in tickers
    assert "AMZN" in tickers

def test_section_aware_chunking(sample_reports):
    """Verifies section-aware chunking preserves section titles."""
    chunker = SectionAwareFinancialChunker(min_chars=20)
    chunks = chunker.chunk_document(sample_reports[0])
    assert len(chunks) > 0
    # Every chunk should contain the company name in brackets
    assert f"[{sample_reports[0].company}" in chunks[0].text

def test_bm25_exact_figure_search(sample_reports):
    """Tests BM25 precision on exact financial figures ($383,285 or iPhone)."""
    chunker = SectionAwareFinancialChunker(min_chars=20)
    all_chunks = []
    for doc in sample_reports:
        all_chunks.extend(chunker.chunk_document(doc))

    bm25 = BM25OkapiRetriever()
    bm25.index_chunks(all_chunks)

    hits = bm25.search("383,285 iPhone", top_k=3)
    assert len(hits) > 0
    assert "Apple" in hits[0]["chunk"].company

def test_hybrid_retrieval_and_rrf(sample_reports):
    """Verifies hybrid Reciprocal Rank Fusion returns top hits."""
    chunker = SectionAwareFinancialChunker(min_chars=20)
    all_chunks = []
    for doc in sample_reports:
        all_chunks.extend(chunker.chunk_document(doc))

    embedder = FinancialEmbeddingEngine(prefer_sentence_transformer=False)
    vec_store = FinancialVectorStore()
    vectors = embedder.embed_texts([c.text for c in all_chunks])
    vec_store.add_chunks(all_chunks, vectors)

    bm25 = BM25OkapiRetriever()
    bm25.index_chunks(all_chunks)

    hybrid = HybridFinancialRetriever(vec_store, bm25, embedder)
    hits = hybrid.retrieve("Microsoft Intelligent Cloud Azure", top_k=3)
    assert len(hits) > 0
    assert any("Microsoft" in h["chunk"].company for h in hits)

def test_refusal_guardrail_on_adversarial_query():
    """Verifies that queries about out-of-corpus topics trigger refusal."""
    pipeline = FinancialRAGPipeline(prefer_neural=False)
    pipeline.load_and_index(REPORTS_DIR)

    result = pipeline.query("What was Apple's total profit from passenger airlines in 2023?")
    assert result["is_refusal"] is True
    assert "INSUFFICIENT_EVIDENCE" in result["answer"]

def test_pipeline_citation_generation():
    """Verifies that successful queries include structured citations."""
    pipeline = FinancialRAGPipeline(prefer_neural=False)
    pipeline.load_and_index(REPORTS_DIR)

    result = pipeline.query("What was Tesla's gross margin and revenue in 2023?")
    assert result["is_refusal"] is False
    assert len(result["citations"]) > 0
    assert any("Tesla" in c for c in result["citations"])
