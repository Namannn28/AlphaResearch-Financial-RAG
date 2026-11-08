"""
Configuration settings for the Financial Research Assistant RAG Pipeline.
Centralizes all file paths, model identifiers, chunking sizes, and safety thresholds.
"""

from dataclasses import dataclass
from pathlib import Path
import os

# Base directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = DATA_DIR / "financial_reports"
EVAL_DIR = DATA_DIR / "evaluation"
EVAL_DATASET_PATH = EVAL_DIR / "financial_eval_dataset.json"
BENCHMARK_RESULTS_PATH = EVAL_DIR / "financial_benchmark_results.json"

@dataclass
class FinancialRAGConfig:
    """Hyperparameters and operational configuration for Financial RAG."""

    # Chunking Hyperparameters
    chunk_size_tokens: int = 350
    chunk_overlap_tokens: int = 50
    min_chunk_length_chars: int = 30

    # Dense Embedding Model (Lightweight 384-dimensional model)
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # Cross-Encoder Reranker Model
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Hybrid Retrieval (Reciprocal Rank Fusion) parameters
    retrieval_top_k: int = 12       # Initial candidates from BM25 & Dense
    rerank_top_k: int = 3          # Final chunks sent to LLM context
    rrf_constant: int = 60          # Standard RRF smoothing constant

    # Safety Refusal Guardrail Threshold
    # If the top reranked confidence score is below this, trigger explicit refusal
    confidence_threshold: float = 0.28

    # LLM Settings
    default_provider: str = "groq"  # "groq" (default free), "mistral", or "offline"
    groq_model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.1
    max_tokens: int = 600

    # Environment variables for API keys
    mistral_api_key: str = os.getenv("MISTRAL_API_KEY", "")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")

# Default instance
config = FinancialRAGConfig()
