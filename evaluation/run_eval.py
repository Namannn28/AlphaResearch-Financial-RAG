"""
Comprehensive Quantitative Evaluation Harness for CandidateFit-RAG.

Executes comparative benchmarks across:
1. Chunking Strategies: Fixed-Window vs Section-Aware Semantic Chunking
2. Retrieval Modes: Dense-only vs BM25-only vs Hybrid RRF
3. Re-ranking: Baseline (No Rerank) vs Cross-Encoder Re-ranked
4. Guardrail Safety: Out-of-Corpus Refusal Accuracy

Outputs exact metrics (Hit Rate@K, MRR@K, Latency, Cost) suitable for README and resumes.
"""

from typing import List, Dict, Any
import json
import time
import sys
from pathlib import Path

# Add project root to sys.path so script can be invoked from any directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from tabulate import tabulate

from src.config import EVAL_DATASET_PATH, DATA_DIR
from src.pipeline import CandidateFitRAGPipeline
from evaluation.metrics import (
    compute_hit_rate_at_k,
    compute_reciprocal_rank,
    compute_precision_at_k,
    evaluate_refusal
)


def load_eval_dataset(eval_path: Path = None) -> List[Dict[str, Any]]:
    path = eval_path or EVAL_DATASET_PATH
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_single_experiment(
    eval_dataset: List[Dict[str, Any]],
    chunking_strategy: str = "section_aware",
    retrieval_mode: str = "hybrid",
    enable_reranking: bool = True
) -> Dict[str, Any]:
    """
    Evaluates one specific architectural configuration across all labeled queries.
    """
    pipeline = CandidateFitRAGPipeline(
        chunking_strategy=chunking_strategy,
        prefer_sentence_transformer=True
    )
    pipeline.build_index()

    answerable_items = [item for item in eval_dataset if item["is_answerable"]]
    unanswerable_items = [item for item in eval_dataset if not item["is_answerable"]]

    hit_at_1_list = []
    hit_at_3_list = []
    hit_at_5_list = []
    mrr_list = []
    precision_at_3_list = []
    latency_list = []

    # 1. Evaluate answerable retrieval quality
    for item in answerable_items:
        query = item["query"]
        target_cand = item["target_candidate"]
        keywords = item.get("ground_truth_keywords", [])

        start_time = time.perf_counter()
        # Retrieve chunks
        raw_results = pipeline.retriever.retrieve(
            query=query,
            top_k=pipeline.vector_store.get_total_chunks(),
            mode=retrieval_mode
        )

        if enable_reranking and raw_results:
            reranked = pipeline.reranker.rerank(
                query=query,
                initial_results=raw_results[:6],
                top_n=5
            )
            retrieved_chunks = [r.chunk for r in reranked]
        else:
            retrieved_chunks = [r.chunk for r in raw_results[:5]]

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        latency_list.append(latency_ms)

        hit_1 = compute_hit_rate_at_k(retrieved_chunks, target_cand, keywords, k=1)
        hit_3 = compute_hit_rate_at_k(retrieved_chunks, target_cand, keywords, k=3)
        hit_5 = compute_hit_rate_at_k(retrieved_chunks, target_cand, keywords, k=5)
        rr = compute_reciprocal_rank(retrieved_chunks, target_cand, keywords, max_k=5)
        p_3 = compute_precision_at_k(retrieved_chunks, target_cand, keywords, k=3)

        hit_at_1_list.append(hit_1)
        hit_at_3_list.append(hit_3)
        hit_at_5_list.append(hit_5)
        mrr_list.append(rr)
        precision_at_3_list.append(p_3)

    # 2. Evaluate refusal safety on unanswerable queries
    refusal_correct_count = 0
    for item in unanswerable_items:
        response = pipeline.query(
            question=item["query"],
            retrieval_mode=retrieval_mode,
            enable_reranking=enable_reranking
        )
        if evaluate_refusal(is_answerable=False, response=response):
            refusal_correct_count += 1

    refusal_accuracy = (refusal_correct_count / max(len(unanswerable_items), 1)) * 100.0

    return {
        "chunking_strategy": chunking_strategy,
        "retrieval_mode": retrieval_mode,
        "reranking_enabled": enable_reranking,
        "eval_samples_answerable": len(answerable_items),
        "eval_samples_unanswerable": len(unanswerable_items),
        "hit_rate_at_1": float(np.mean(hit_at_1_list)),
        "hit_rate_at_3": float(np.mean(hit_at_3_list)),
        "hit_rate_at_5": float(np.mean(hit_at_5_list)),
        "mrr_at_5": float(np.mean(mrr_list)),
        "precision_at_3": float(np.mean(precision_at_3_list)),
        "refusal_accuracy": float(refusal_accuracy),
        "avg_retrieval_latency_ms": float(np.mean(latency_list))
    }


def run_full_benchmark_suite() -> Dict[str, Any]:
    """Runs comparative experiments matching the candidate evaluation protocol."""
    eval_dataset = load_eval_dataset()

    print(f"\n=======================================================")
    print(f" CandidateFit-RAG: Benchmark Evaluation Suite")
    print(f" Total Evaluation Questions: {len(eval_dataset)}")
    print(f"=======================================================\n")

    experiments = [
        # Baseline: Fixed window, pure dense, no reranking
        {
            "name": "Fixed-Window + Dense (Tutorial Baseline)",
            "chunking": "fixed_window",
            "mode": "dense_only",
            "rerank": False
        },
        # Experiment 2: Fixed window + Hybrid RRF
        {
            "name": "Fixed-Window + Hybrid RRF",
            "chunking": "fixed_window",
            "mode": "hybrid",
            "rerank": False
        },
        # Experiment 3: Section-Aware + Hybrid RRF (No Rerank)
        {
            "name": "Section-Aware + Hybrid RRF (No Rerank)",
            "chunking": "section_aware",
            "mode": "hybrid",
            "rerank": False
        },
        # Full Production Pipeline: Section-Aware + Hybrid RRF + Cross-Encoder Rerank
        {
            "name": "Section-Aware + Hybrid RRF + Reranker (Production)",
            "chunking": "section_aware",
            "mode": "hybrid",
            "rerank": True
        }
    ]

    benchmark_records = []
    table_rows = []

    for exp in experiments:
        print(f"Running experiment: {exp['name']}...")
        result = run_single_experiment(
            eval_dataset=eval_dataset,
            chunking_strategy=exp["chunking"],
            retrieval_mode=exp["mode"],
            enable_reranking=exp["rerank"]
        )
        result["configuration_name"] = exp["name"]
        benchmark_records.append(result)

        table_rows.append([
            exp["name"],
            f"{result['hit_rate_at_1'] * 100:.1f}%",
            f"{result['hit_rate_at_3'] * 100:.1f}%",
            f"{result['mrr_at_5']:.3f}",
            f"{result['precision_at_3'] * 100:.1f}%",
            f"{result['refusal_accuracy']:.1f}%",
            f"{result['avg_retrieval_latency_ms']:.1f}ms"
        ])

    headers = [
        "Architecture Configuration",
        "Hit@1",
        "Hit@3 (Recall@3)",
        "MRR@5",
        "Precision@3",
        "Refusal Acc",
        "Latency"
    ]

    print("\n" + tabulate(table_rows, headers=headers, tablefmt="github") + "\n")

    # Save results to JSON for Streamlit UI and README embedding
    output_path = DATA_DIR / "evaluation" / "benchmark_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_records, f, indent=2)

    print(f"Benchmark results saved to: {output_path}")
    return {"results": benchmark_records, "summary_table": table_rows}


if __name__ == "__main__":
    run_full_benchmark_suite()
