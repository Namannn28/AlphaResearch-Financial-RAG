"""
End-to-End Pipeline Orchestrator for CandidateFit-RAG.

Coordinates document ingestion, chunking, embedding, vector & lexical indexing,
hybrid retrieval, cross-encoder re-ranking, and grounded generation.
"""

from typing import List, Dict, Any, Optional
import time
import logging

from src.config import RESUMES_DIR, retrieval_config
from src.ingestion import load_corpus_from_directory, Document
from src.chunking import build_chunks, Chunk
from src.embeddings import get_embedder, BaseEmbedder
from src.vector_store import InMemoryVectorStore
from src.lexical_search import BM25LexicalSearcher
from src.hybrid_retriever import HybridRetriever, RetrievalResult
from src.reranker import CrossEncoderReranker, RerankResult
from src.generator import GroundedGenerator, GenerationResponse

logger = logging.getLogger(__name__)


class CandidateFitRAGPipeline:
    """
    Production-structured RAG pipeline for candidate resume & job fit intelligence.
    """
    def __init__(
        self,
        chunking_strategy: str = "section_aware",
        prefer_sentence_transformer: bool = True
    ):
        self.chunking_strategy = chunking_strategy
        self.documents: List[Document] = []
        self.chunks: List[Chunk] = []

        # Core subsystems
        self.embedder: BaseEmbedder = get_embedder(prefer_sentence_transformer=prefer_sentence_transformer)
        self.vector_store = InMemoryVectorStore()
        self.lexical_searcher = BM25LexicalSearcher()
        self.retriever = HybridRetriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
            lexical_searcher=self.lexical_searcher
        )
        self.reranker = CrossEncoderReranker(enable_model=prefer_sentence_transformer)
        self.generator = GroundedGenerator()

        self.is_indexed = False

    def build_index(self, resumes_dir=None):
        """
        Loads candidate resumes, segments into chunks, generates embeddings,
        and initializes both vector and BM25 indices.
        """
        target_dir = resumes_dir or RESUMES_DIR
        logger.info("Ingesting candidate resumes from %s...", target_dir)
        self.documents = load_corpus_from_directory(target_dir)

        logger.info("Chunking %d documents using '%s' strategy...", len(self.documents), self.chunking_strategy)
        self.chunks = build_chunks(self.documents, strategy=self.chunking_strategy)

        logger.info("Generating embeddings for %d chunks (dim=%d)...", len(self.chunks), self.embedder.dimension)
        chunk_texts = [c.text for c in self.chunks]
        embeddings = self.embedder.embed_texts(chunk_texts)

        # Index into vector store
        self.vector_store.clear()
        self.vector_store.add_chunks(self.chunks, embeddings)

        # Index into BM25 lexical ranker
        self.lexical_searcher.index_chunks(self.chunks)

        self.is_indexed = True
        logger.info("CandidateFit-RAG index built successfully (%d chunks).", len(self.chunks))

    def query(
        self,
        question: str,
        candidate_name_filter: Optional[str] = None,
        retrieval_mode: str = "hybrid",
        enable_reranking: bool = True
    ) -> GenerationResponse:
        """
        Executes an end-to-end question answering query.
        
        Steps:
        1. Retrieval (Hybrid / Dense / BM25) -> top-K
        2. Re-Ranking (Cross-Encoder) -> top-N
        3. Generation & Refusal Guardrail -> GenerationResponse
        """
        if not self.is_indexed:
            self.build_index()

        retrieval_start = time.perf_counter()

        # Step 1: Initial Retrieval
        initial_top_k = retrieval_config.initial_top_k
        raw_results = self.retriever.retrieve(
            query=question,
            top_k=initial_top_k,
            mode=retrieval_mode,
            candidate_name_filter=candidate_name_filter
        )

        # Step 2: Re-ranking
        if enable_reranking and raw_results:
            reranked_results = self.reranker.rerank(
                query=question,
                initial_results=raw_results,
                top_n=retrieval_config.rerank_top_n
            )
        else:
            # Wrap raw results without rank change
            reranked_results = [
                RerankResult(
                    chunk=res.chunk,
                    initial_rank=i + 1,
                    reranked_rank=i + 1,
                    initial_score=res.score,
                    rerank_score=res.score,
                    rank_lift=0
                )
                for i, res in enumerate(raw_results[:retrieval_config.rerank_top_n])
            ]

        retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000.0

        # Step 3: Grounded Answer Generation
        response = self.generator.generate(
            query=question,
            reranked_results=reranked_results,
            retrieval_latency_ms=retrieval_latency_ms
        )

        return response

    def match_candidate_against_job(
        self,
        job_description_text: str,
        candidate_name: str
    ) -> Dict[str, Any]:
        """
        Extracts key requirements from a job description and audits candidate alignment.
        Returns matching evidence and missing skill gaps.
        """
        if not self.is_indexed:
            self.build_index()

        # Extract requirement lines
        req_lines = [
            line.strip().lstrip("-*• ")
            for line in job_description_text.split("\n")
            if line.strip() and not line.startswith("#") and len(line.strip()) > 15
        ][:6]

        evaluations = []
        overall_scores = []

        for req in req_lines:
            resp = self.query(
                question=f"Does {candidate_name} have experience with: {req}?",
                candidate_name_filter=candidate_name,
                retrieval_mode="hybrid"
            )
            has_evidence = not resp.is_refusal and resp.confidence_score >= retrieval_config.confidence_refusal_threshold
            evaluations.append({
                "requirement": req,
                "has_evidence": has_evidence,
                "confidence": resp.confidence_score,
                "answer": resp.answer,
                "citations": resp.citations
            })
            if has_evidence:
                overall_scores.append(resp.confidence_score)

        alignment_percentage = (len(overall_scores) / max(len(req_lines), 1)) * 100.0

        return {
            "candidate_name": candidate_name,
            "requirements_evaluated": len(req_lines),
            "requirements_met": len(overall_scores),
            "alignment_percentage": round(alignment_percentage, 1),
            "evaluations": evaluations
        }

    def get_corpus_statistics(self) -> Dict[str, Any]:
        """Provides operational metrics regarding the active index."""
        candidate_names = sorted(list({d.metadata.get("candidate_name") for d in self.documents}))
        return {
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunks),
            "chunking_strategy": self.chunking_strategy,
            "embedding_dimension": self.embedder.dimension,
            "candidate_names": candidate_names
        }
