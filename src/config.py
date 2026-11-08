"""
Configuration module for CandidateFit-RAG.

This file centralizes all tunable parameters, file paths, model configurations,
and evaluation thresholds. By defining these centrally, any engineer or recruiter
reviewing the codebase can immediately observe the technical trade-offs chosen.
"""

from dataclasses import dataclass
from pathlib import Path
import os

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RESUMES_DIR = DATA_DIR / "resumes"
JOB_DESCRIPTIONS_DIR = DATA_DIR / "job_descriptions"
EVALUATION_DIR = DATA_DIR / "evaluation"
EVAL_DATASET_PATH = EVALUATION_DIR / "eval_dataset.json"


@dataclass(frozen=True)
class IngestionConfig:
    """Settings for loading and preprocessing candidate documents."""
    allowed_extensions: tuple = (".md", ".txt", ".pdf")
    strip_boilerplate: bool = True
    normalize_whitespace: bool = True


@dataclass(frozen=True)
class ChunkingConfig:
    """
    Settings comparing two retrieval segmentation philosophies:
    1. Fixed-Window: standard sliding window approach (common baseline).
    2. Section-Aware: boundary-aware semantic chunking based on resume structure
       (e.g., Experience, Projects, Technical Skills, Education).
    """
    # Strategy A: Fixed-window chunking
    fixed_chunk_size_words: int = 140
    fixed_chunk_overlap_words: int = 30

    # Strategy B: Section-based semantic chunking
    min_section_length_chars: int = 20
    max_section_length_chars: int = 1500


@dataclass(frozen=True)
class RetrievalConfig:
    """
    Retrieval, Re-ranking, and Guardrail thresholds.
    """
    # Primary dense embedding model name (fast, 384 dimensions, permissive Apache 2.0 license)
    dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dense_dimension: int = 384

    # Top-K candidate chunks fetched during initial bi-encoder retrieval
    initial_top_k: int = 6

    # Top-N chunks retained after cross-encoder re-ranking
    rerank_top_n: int = 3

    # RRF (Reciprocal Rank Fusion) constant for combining dense and lexical ranks
    rrf_constant: int = 60

    # Strict confidence threshold for hallucination refusal.
    # If the top retrieval similarity is below this value, the system refuses to answer.
    # This prevents the LLM from fabricating candidate claims that do not exist.
    confidence_refusal_threshold: float = 0.32


@dataclass(frozen=True)
class GenerationConfig:
    """Parameters for grounded answer synthesis."""
    # LLM temperature set to 0.0 for deterministic, strictly grounded factual responses
    temperature: float = 0.0
    max_tokens: int = 500
    
    # Cost profiling constants (estimated per 1,000 tokens for lightweight cloud models)
    estimated_input_cost_per_1k_tokens: float = 0.00015
    estimated_output_cost_per_1k_tokens: float = 0.00060


# Global configuration instances
ingestion_config = IngestionConfig()
chunking_config = ChunkingConfig()
retrieval_config = RetrievalConfig()
generation_config = GenerationConfig()
