r"""
==============================================================================
RAG EVALUATION & BENCHMARK HARNESS (src/evaluation.py)
==============================================================================
Role in RAG Pipeline:
    Automated evaluation framework that benchmarks retrieval accuracy and
    hallucination refusal across four distinct architectures:
    1. Pure Dense Vector Search (SentenceTransformers MiniLM)
    2. Pure BM25 Lexical Search (Okapi BM25 keyword matching)
    3. Hybrid Search (Reciprocal Rank Fusion - RRF)
    4. Hybrid + Cross-Encoder Reranker & Guardrail (Full Production Pipeline)

Key Metrics Computed:
    - Recall@1, Recall@3, Recall@5: Percentage of questions where gold factual
      source chunks appeared in top-K candidates.
    - MRR (Mean Reciprocal Rank): Measures rank position quality (1/rank).
    - Average Latency (ms): Wall-clock retrieval speed per inquiry.
    - Refusal Precision / Safety: Percentage of adversarial out-of-corpus
      trap queries correctly rejected below confidence threshold (tau = 0.28).

Usage:
    python src/evaluation.py
==============================================================================
"""

import sys
from pathlib import Path

# Add project root to path for direct module execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import time
from typing import List, Dict, Any
import numpy as np

from src.config import config, REPORTS_DIR, EVAL_DATASET_PATH, BENCHMARK_RESULTS_PATH
from src.pipeline import FinancialRAGPipeline
from src.chunking import FinancialChunk


def is_chunk_match(chunk: FinancialChunk, gold_keywords: List[str]) -> bool:
    """Checks if a retrieved chunk satisfies gold keywords."""
    text_lower = chunk.text.lower()
    matches = sum(1 for kw in gold_keywords if kw.lower() in text_lower)
    return matches >= 1


class RAGEvaluator:
    """Automated benchmark harness for evaluating Financial RAG pipelines."""

    def __init__(self, dataset_path: Path = EVAL_DATASET_PATH):
        self.dataset_path = dataset_path
        with open(dataset_path, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)

        print(f"[Init] Loaded {len(self.dataset)} evaluation items from {dataset_path.name}")
        self.factual_items = [item for item in self.dataset if not item.get("is_adversarial", False)]
        self.adversarial_items = [item for item in self.dataset if item.get("is_adversarial", False)]

        print(f"       * Factual Questions: {len(self.factual_items)}")
        print(f"       * Adversarial Trap Questions: {len(self.adversarial_items)}")

    def run_benchmark(self, pipeline: FinancialRAGPipeline) -> Dict[str, Any]:
        """Runs the complete benchmark across all four retrieval architectures."""
        modes = [
            {"name": "Dense Vector Search", "use_hybrid": False, "use_reranker": False},
            {"name": "BM25 Lexical Search", "use_hybrid": "bm25_only", "use_reranker": False},
            {"name": "Hybrid (BM25 + Dense RRF)", "use_hybrid": True, "use_reranker": False},
            {"name": "Hybrid + Cross-Encoder Reranker", "use_hybrid": True, "use_reranker": True}
        ]

        benchmark_results = {}

        for mode in modes:
            mode_name = mode["name"]
            print(f"\n========================================================")
            print(f" Benchmarking Engine: {mode_name}")
            print(f"========================================================")

            # 1. Evaluate Factual Retrieval Accuracy
            r1_hits = 0
            r3_hits = 0
            r5_hits = 0
            reciprocal_ranks = []
            latencies = []

            for item in self.factual_items:
                query = item["question"]
                gold_kws = item["gold_keywords"]
                target_company = item["company"] if item["company"] != "Cross-Company" else None

                t0 = time.perf_counter()

                # Execute retrieval according to tested mode
                if mode["use_hybrid"] == "bm25_only":
                    hits = pipeline.bm25_retriever.search(query, top_k=10, filter_company=target_company)
                elif mode["use_hybrid"] is True:
                    initial_hits = pipeline.hybrid_retriever.retrieve(query, top_k=10, filter_company=target_company)
                    if mode["use_reranker"]:
                        hits = pipeline.reranker.rerank(query, initial_hits, top_k=5)
                    else:
                        hits = initial_hits[:5]
                else:
                    # Pure Dense
                    q_vec = pipeline.embedding_engine.embed_query(query)
                    hits = pipeline.vector_store.similarity_search(q_vec, top_k=10, filter_company=target_company)

                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                latencies.append(elapsed_ms)

                # Score ranks
                found_rank = None
                for rank_idx, hit in enumerate(hits[:5], start=1):
                    chunk = hit["chunk"]
                    if is_chunk_match(chunk, gold_kws):
                        found_rank = rank_idx
                        break

                if found_rank == 1:
                    r1_hits += 1
                if found_rank is not None and found_rank <= 3:
                    r3_hits += 1
                if found_rank is not None and found_rank <= 5:
                    r5_hits += 1

                if found_rank is not None:
                    reciprocal_ranks.append(1.0 / found_rank)
                else:
                    reciprocal_ranks.append(0.0)

            total_factual = len(self.factual_items)
            recall_1 = r1_hits / total_factual
            recall_3 = r3_hits / total_factual
            recall_5 = r5_hits / total_factual
            mrr = float(np.mean(reciprocal_ranks))
            avg_latency = float(np.mean(latencies))

            # 2. Evaluate Adversarial Refusal Precision
            refused_count = 0
            for item in self.adversarial_items:
                query = item["question"]
                target_company = item["company"]

                if mode["use_reranker"]:
                    initial_hits = pipeline.hybrid_retriever.retrieve(query, top_k=10, filter_company=target_company)
                    reranked = pipeline.reranker.rerank(query, initial_hits, top_k=3)
                    top_conf = reranked[0]["score"] if reranked else 0.0
                    if top_conf < config.confidence_threshold:
                        refused_count += 1
                else:
                    # Without reranker guardrail, pure retrievers always return chunks without refusal capability
                    refused_count += 0

            refusal_precision = refused_count / max(1, len(self.adversarial_items))

            benchmark_results[mode_name] = {
                "recall@1": round(recall_1, 4),
                "recall@3": round(recall_3, 4),
                "recall@5": round(recall_5, 4),
                "mrr": round(mrr, 4),
                "avg_latency_ms": round(avg_latency, 2),
                "refusal_precision": round(refusal_precision, 4)
            }

            print(f"  Recall@1:           {recall_1 * 100:.1f}%")
            print(f"  Recall@3:           {recall_3 * 100:.1f}%")
            print(f"  Recall@5:           {recall_5 * 100:.1f}%")
            print(f"  MRR:                {mrr:.4f}")
            print(f"  Avg Latency:        {avg_latency:.1f} ms")
            print(f"  Refusal Precision:  {refusal_precision * 100:.1f}%")

        # Save results to disk
        BENCHMARK_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(BENCHMARK_RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(benchmark_results, f, indent=2)

        print(f"\n[OK] Benchmark results saved to {BENCHMARK_RESULTS_PATH}")
        return benchmark_results

    @staticmethod
    def print_markdown_summary(results: Dict[str, Any]):
        """Prints formatted Markdown table for README and Cheat Sheet."""
        print("\n" + "=" * 80)
        print("🏆 ALPHA RESEARCH — COMPARATIVE RAG BENCHMARK RESULTS")
        print("=" * 80)
        headers = ["Architecture Mode", "Recall@1", "Recall@3", "Recall@5", "MRR", "Latency (ms)", "Refusal Safety"]
        print(f"| {' | '.join(headers)} |")
        print(f"|{'-' * 32}|{'-' * 10}|{'-' * 10}|{'-' * 10}|{'-' * 8}|{'-' * 14}|{'-' * 16}|")

        for mode_name, metrics in results.items():
            r1 = f"{metrics['recall@1'] * 100:.1f}%"
            r3 = f"{metrics['recall@3'] * 100:.1f}%"
            r5 = f"{metrics['recall@5'] * 100:.1f}%"
            mrr = f"{metrics['mrr']:.3f}"
            lat = f"{metrics['avg_latency_ms']:.1f} ms"
            refusal = f"{metrics['refusal_precision'] * 100:.1f}%"
            print(f"| {mode_name:<30} | {r1:>8} | {r3:>8} | {r5:>8} | {mrr:>6} | {lat:>12} | {refusal:>14} |")
        print("=" * 80)


if __name__ == "__main__":
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 80)
    print("STARTING ALPHA RESEARCH RAG BENCHMARK EVALUATION")
    print("=" * 80)

    # Initialize Base Pipeline
    pipe = FinancialRAGPipeline(use_hybrid=True, use_reranker=True)
    pipe.load_and_index(REPORTS_DIR)

    evaluator = RAGEvaluator()
    res = evaluator.run_benchmark(pipe)
    evaluator.print_markdown_summary(res)
