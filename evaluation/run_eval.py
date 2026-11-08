"""
Evaluation Benchmark Runner for Financial Research Assistant RAG.
Runs comparative benchmark across 4 system configurations:
1. Baseline: Fixed-Window Chunking + Dense Retrieval Only
2. Stage 1: Section-Aware Chunking + Dense Retrieval Only
3. Stage 2: Section-Aware Chunking + Hybrid Retrieval (BM25 + Dense RRF)
4. Production: Section-Aware + Hybrid + Cross-Encoder Reranker
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List

from src.config import EVAL_DATASET_PATH, BENCHMARK_RESULTS_PATH, REPORTS_DIR
from src.pipeline import FinancialRAGPipeline
from evaluation.metrics import RAGEvaluator

def load_eval_dataset() -> List[Dict[str, Any]]:
    with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def run_evaluation_suite() -> Dict[str, Any]:
    dataset = load_eval_dataset()
    in_corpus_items = [item for item in dataset if item["is_in_corpus"]]
    out_of_corpus_items = [item for item in dataset if not item["is_in_corpus"]]

    configurations = [
        {
            "name": "Baseline (Fixed-Window + Dense)",
            "use_section_chunker": False,
            "use_hybrid": False,
            "use_reranker": False
        },
        {
            "name": "Section-Aware + Dense",
            "use_section_chunker": True,
            "use_hybrid": False,
            "use_reranker": False
        },
        {
            "name": "Section-Aware + Hybrid (BM25+Dense)",
            "use_section_chunker": True,
            "use_hybrid": True,
            "use_reranker": False
        },
        {
            "name": "Production (Hybrid + Cross-Encoder)",
            "use_section_chunker": True,
            "use_hybrid": True,
            "use_reranker": True
        }
    ]

    all_results = {}

    print(f"\n=======================================================")
    print(f"Starting Financial RAG Benchmark ({len(dataset)} Questions)")
    print(f"In-Corpus: {len(in_corpus_items)} | Out-of-Corpus Traps: {len(out_of_corpus_items)}")
    print(f"=======================================================\n")

    for cfg in configurations:
        cfg_name = cfg["name"]
        print(f"Evaluating Configuration: {cfg_name}...")

        pipeline = FinancialRAGPipeline(
            use_section_chunker=cfg["use_section_chunker"],
            use_hybrid=cfg["use_hybrid"],
            use_reranker=cfg["use_reranker"],
            prefer_neural=True
        )
        pipeline.load_and_index(REPORTS_DIR)

        hit_1_list = []
        hit_3_list = []
        mrr_5_list = []
        p_3_list = []
        kw_recall_list = []
        latencies = []

        # Evaluate on in-corpus factual queries
        for item in in_corpus_items:
            q = item["question"]
            target = item["target_company"]
            kws = item.get("expected_keywords", [])

            t0 = time.perf_counter()
            res = pipeline.query(q, top_k=5)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat_ms)

            retrieved = res["retrieved_chunks"]
            hit_1_list.append(RAGEvaluator.calculate_hit_rate(retrieved, target, k=1))
            hit_3_list.append(RAGEvaluator.calculate_hit_rate(retrieved, target, k=3))
            mrr_5_list.append(RAGEvaluator.calculate_reciprocal_rank(retrieved, target, max_k=5))
            p_3_list.append(RAGEvaluator.calculate_precision_at_k(retrieved, target, k=3))
            kw_recall_list.append(RAGEvaluator.check_keyword_recall(retrieved, kws, k=3))

        # Evaluate refusal accuracy on out-of-corpus adversarial traps
        refusals = 0
        for item in out_of_corpus_items:
            q = item["question"]
            res = pipeline.query(q, top_k=3)
            if res["is_refusal"]:
                refusals += 1

        refusal_accuracy = (refusals / len(out_of_corpus_items)) if out_of_corpus_items else 1.0

        metrics = {
            "hit_at_1": round(sum(hit_1_list) / len(hit_1_list), 4),
            "hit_at_3": round(sum(hit_3_list) / len(hit_3_list), 4),
            "mrr_at_5": round(sum(mrr_5_list) / len(mrr_5_list), 4),
            "precision_at_3": round(sum(p_3_list) / len(p_3_list), 4),
            "keyword_recall_at_3": round(sum(kw_recall_list) / len(kw_recall_list), 4),
            "refusal_accuracy": round(refusal_accuracy, 4),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2)
        }

        all_results[cfg_name] = metrics
        print(f"  -> Hit@1: {metrics['hit_at_1']*100:.1f}% | Recall@3: {metrics['hit_at_3']*100:.1f}% | MRR@5: {metrics['mrr_at_5']:.3f} | Refusal: {metrics['refusal_accuracy']*100:.1f}% | Latency: {metrics['avg_latency_ms']}ms")

    # Serialize to JSON
    with open(BENCHMARK_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nBenchmark complete. Saved results to {BENCHMARK_RESULTS_PATH}\n")
    return all_results

if __name__ == "__main__":
    run_evaluation_suite()
