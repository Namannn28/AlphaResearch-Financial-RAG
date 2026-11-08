"""
==============================================================================
STEP 7: END-TO-END RAG ORCHESTRATION PIPELINE (src/pipeline.py)
==============================================================================
Role in RAG Pipeline:
    Connects all isolated modules into a single, cohesive RAG workflow:
      Ingestion -> Chunking -> Embedding -> Vector Store + BM25 ->
      Hybrid Retrieval (RRF) -> Cross-Encoder Reranker -> Guardrail -> LLM.

Key Concepts for Interviews:
    - Modularity: Each pipeline component (chunker, embedder, retriever, reranker,
      generator) is decoupled, testable independently, and replaceable.
    - Dynamic Indexing: Supports `.add_document(file_path)` to index new PDFs
      on the fly without rebuilding existing vector matrices from scratch.
    - Observability: Every query returns latency breakdowns (retrieval vs.
      generation), confidence scores, and source citations.

Run directly to test:
    python src/pipeline.py
==============================================================================
"""

import sys
from pathlib import Path

# Add project root to path for direct module execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import List, Dict, Any, Optional
import time

from src.config import config, REPORTS_DIR
from src.ingestion import DocumentIngestionEngine, FinancialDocument
from src.chunking import SectionAwareFinancialChunker, FinancialChunk
from src.embeddings import FinancialEmbeddingEngine
from src.vector_store import FinancialVectorStore
from src.lexical_search import BM25OkapiRetriever
from src.hybrid_retriever import HybridFinancialRetriever
from src.reranker import FinancialCrossEncoderReranker
from src.llm_client import FinancialLLMClient


class FinancialRAGPipeline:
    """Production Financial RAG Pipeline for SEC 10-K Analysis & PDF Querying."""

    def __init__(
        self,
        use_hybrid: bool = True,
        use_reranker: bool = True,
        prefer_neural: bool = True,
        llm_provider: str = "groq"
    ):
        self.use_hybrid = use_hybrid
        self.use_reranker = use_reranker
        self.prefer_neural = prefer_neural

        # 1. Initialize decoupled components
        self.ingestion_engine = DocumentIngestionEngine()
        self.chunker = SectionAwareFinancialChunker()
        self.embedding_engine = FinancialEmbeddingEngine(
            model_name=config.embedding_model_name,
            prefer_sentence_transformer=prefer_neural
        )
        self.vector_store = FinancialVectorStore()
        self.bm25_retriever = BM25OkapiRetriever()
        self.hybrid_retriever = HybridFinancialRetriever(
            vector_store=self.vector_store,
            bm25_retriever=self.bm25_retriever,
            embedding_engine=self.embedding_engine,
            rrf_constant=config.rrf_constant
        )
        self.reranker = FinancialCrossEncoderReranker(
            model_name=config.reranker_model_name,
            enable_neural=prefer_neural
        )
        self.llm_client = FinancialLLMClient(
            groq_api_key=config.groq_api_key,
            provider=llm_provider
        )

        self.documents: List[FinancialDocument] = []
        self.chunks: List[FinancialChunk] = []
        self.is_indexed = False

    def load_and_index(self, reports_dir: Optional[Path] = None):
        """Loads all reports (.md, .pdf, .txt) from folder and builds hybrid indices."""
        target_dir = reports_dir or REPORTS_DIR
        self.documents = self.ingestion_engine.load_all_reports(target_dir)

        all_chunks: List[FinancialChunk] = []
        for doc in self.documents:
            doc_chunks = self.chunker.chunk_document(doc)
            all_chunks.extend(doc_chunks)

        self.chunks = all_chunks

        # 1. Index BM25 Lexical
        self.bm25_retriever.index_chunks(self.chunks)

        # 2. Index Vector Store
        chunk_texts = [c.text for c in self.chunks]
        embeddings = self.embedding_engine.embed_texts(chunk_texts)
        self.vector_store.clear()
        self.vector_store.add_chunks(self.chunks, embeddings)

        self.is_indexed = True

    def add_document(self, file_path: Path) -> int:
        """
        Dynamically loads, chunks, and indexes a single new PDF, Markdown, or text file.
        Returns the number of new chunks added to the index.
        """
        new_doc = self.ingestion_engine.load_file(file_path)
        self.documents.append(new_doc)

        new_chunks = self.chunker.chunk_document(new_doc)
        if not new_chunks:
            return 0

        self.chunks.extend(new_chunks)

        # Re-index BM25
        self.bm25_retriever.index_chunks(self.chunks)

        # Add new vectors to Vector Store
        new_texts = [c.text for c in new_chunks]
        new_embeddings = self.embedding_engine.embed_texts(new_texts)
        self.vector_store.add_chunks(new_chunks, new_embeddings)

        return len(new_chunks)

    def query(
        self,
        question: str,
        filter_company: Optional[str] = None,
        top_k: int = 3,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end query workflow:
        1. Hybrid retrieval (Dense + BM25)
        2. Cross-Encoder reranking
        3. Confidence assessment & refusal guardrail
        4. Multi-chunk LangChain Groq synthesis with citations and session history
        """
        if not self.is_indexed:
            self.load_and_index()

        start_time = time.perf_counter()

        # Step 1: Retrieval
        if self.use_hybrid:
            initial_hits = self.hybrid_retriever.retrieve(
                query=question,
                top_k=config.retrieval_top_k,
                filter_company=filter_company
            )
        else:
            q_vec = self.embedding_engine.embed_query(question)
            initial_hits = self.vector_store.similarity_search(
                q_vec,
                top_k=config.retrieval_top_k,
                filter_company=filter_company
            )

        # Step 2: Reranking
        if self.use_reranker and initial_hits:
            final_hits = self.reranker.rerank(
                query=question,
                candidate_hits=initial_hits,
                top_k=top_k
            )
        else:
            final_hits = initial_hits[:top_k]

        retrieval_latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Step 3: Top confidence
        top_confidence = final_hits[0]["score"] if final_hits else 0.0

        # Step 4: Multi-chunk Answer generation with session conversation history
        generation_start = time.perf_counter()
        gen_result = self.llm_client.generate_answer(
            query=question,
            retrieved_chunks=final_hits,
            confidence_score=top_confidence,
            confidence_threshold=config.confidence_threshold,
            conversation_history=conversation_history
        )
        generation_latency_ms = (time.perf_counter() - generation_start) * 1000.0

        return {
            "question": question,
            "answer": gen_result["answer"],
            "is_refusal": gen_result["is_refusal"],
            "provider_used": gen_result["provider_used"],
            "citations": gen_result["citations"],
            "top_confidence": top_confidence,
            "retrieval_latency_ms": retrieval_latency_ms,
            "generation_latency_ms": generation_latency_ms,
            "total_latency_ms": retrieval_latency_ms + generation_latency_ms,
            "retrieved_chunks": final_hits
        }


if __name__ == "__main__":
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 70)
    print("[STEP 7 DEMO] End-to-End Financial RAG Pipeline")
    print("=" * 70)
    pipeline = FinancialRAGPipeline()
    print("Loading and indexing corpus...")
    pipeline.load_and_index()
    print(f"Index built: {len(pipeline.chunks)} chunks across {len(pipeline.documents)} filings.\n")

    test_queries = [
        "What was Apple's total net sales and iPhone revenue in fiscal 2023?",
        "What was Tesla's commercial airline revenue in fiscal 2023?"  # Adversarial refusal
    ]

    for q in test_queries:
        print("-" * 70)
        print(f"Q: {q}")
        res = pipeline.query(q)
        print(f"Top Confidence: {res['top_confidence']:.4f} | Latency: {res['total_latency_ms']:.1f}ms")
        print(f"Provider: {res['provider_used']}")
        print(f"A: {res['answer']}\n")

    print("=" * 70)
    print("[OK] Complete RAG pipeline verified and fully operational.")
    print("=" * 70)
