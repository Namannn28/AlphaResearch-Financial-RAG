# System Architecture & Technical Design Specification

> **System Name**: AlphaResearch  
> **Domain**: Financial Document Intelligence & SEC 10-K Retrieval-Augmented Generation  
> **Status**: Production / Evaluated  

---

## 1. Executive Summary

Enterprise financial analysis demands strict quantitative precision, absolute source traceability, and deterministic hallucination prevention. Standard naive RAG architectures—relying solely on dense vector search over arbitrary text chunks—fail in financial contexts due to:
1. **Numerical and Ticker Drift**: Dense embeddings approximate semantic concepts but often conflate closely spaced figures (e.g., $383,285M vs $393,285M) or ticker symbols (`GOOG` vs `GOOGL`).
2. **Table Fragmentation**: Naive fixed-token sliding windows slice financial balance sheets and income tables across arbitrary tokens, severing rows from column headers.
3. **Plausible Hallucinations**: Standard LLM generation eagerly produces speculative responses when queried about non-existent divisions or out-of-corpus metrics.

**AlphaResearch** implements a multi-stage, production-grade retrieval pipeline combining **Section-Aware Financial Chunking**, **Dense + BM25 Hybrid Retrieval** fused via **Reciprocal Rank Fusion (RRF)**, **Cross-Encoder Reranking**, and an **Adversarial Confidence Refusal Guardrail**.

---

## 2. End-to-End System Architecture

```
                  ┌─────────────────────────────────────────┐
                  │ User Query (Terminal CLI / Streamlit UI)│
                  └───────────────────┬─────────────────────┘
                                      │
                         [Query Preprocessor]
                     (Intent & Greeting Detection)
                                      │
                      ┌───────────────┴───────────────┐
                      ▼                               ▼
            ┌──────────────────┐            ┌──────────────────┐
            │ Dense Embedding  │            │  BM25 Lexical    │
            │ MiniLM (384-dim) │            │  Exact Figures   │
            │  Semantic Search │            │ (AAPL, $383.29B) │
            └─────────┬────────┘            └─────────┬────────┘
                      │ Top-12 Hits                   │ Top-12 Hits
                      └───────────────┬───────────────┘
                                      │
                                      ▼
                          ┌───────────────────────┐
                          │ Reciprocal Rank Fusion│
                          │  RRF = 1 / (60 + r)   │
                          └───────────┬───────────┘
                                      │ Top Candidate Chunks
                                      ▼
                          ┌───────────────────────┐
                          │ Cross-Encoder Reranker│
                          │  (Attention Re-score) │
                          └───────────┬───────────┘
                                      │
                         Confidence < Threshold (0.28)?
                                     ├───► YES ──► Refusal Guardrail: "INSUFFICIENT_EVIDENCE"
                                     │
                                     └───► NO
                                            │ Top Grounded Chunks
                                            ▼
                            ┌───────────────────────────────┐
                            │   LangChain ChatGroq LLM      │
                            │  (Offline Extractive Fallback)│
                            └───────────────┬───────────────┘
                                            │
                                            ▼
                           Final Answer with Source Citations
```

---

## 3. Component Deep Dive

### 3.1 Ingestion & Section-Aware Chunking (`src/ingestion.py`, `src/chunking.py`)
- **Multi-Format Ingestion**: Ingests Markdown disclosures and native PDFs (`pypdf`), extracting page indices and heading hierarchies.
- **Header Enrichment**: Every chunk is prepended with contextual metadata:
  ```
  [Company (Ticker) | Section Title | Page X]
  ```
  This prevents "orphan chunks" where financial rows appear divorced from their respective company and reporting year.
- **Adaptive Windowing**: Groups lines into 200–250 word semantic units while preventing table row truncation.

### 3.2 Dual-Channel Retrieval (`src/vector_store.py`, `src/lexical_search.py`)
- **Dense Vector Channel**: Uses `sentence-transformers/all-MiniLM-L6-v2` generating 384-dimensional unit-normalized embeddings. Inner dot-product executes sub-millisecond cosine similarity search across in-memory matrix representations.
- **Lexical Channel (Okapi BM25)**: Maintains an inverted index with custom financial tokenization preserving currency symbols (`$`), decimals (`44.1%`), and stock tickers (`MSFT`, `TSLA`).

### 3.3 Reciprocal Rank Fusion (`src/hybrid_retriever.py`)
Dense and lexical candidate lists are fused using rank-based reciprocal aggregation:
$$\text{Score}_{\text{RRF}}(d) = \sum_{m \in \{\text{Dense}, \text{BM25}\}} \frac{w_m}{k + \text{rank}_m(d)}$$
where $k = 60$ is the standard smoothing parameter. This rank-invariant formulation eliminates the need to normalize disparate cosine and BM25 score distributions.

### 3.4 Cross-Encoder Reranking & Safety Guardrail (`src/reranker.py`)
- **Joint Attention**: Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` to compute deep cross-attention over `[CLS] Query [SEP] Passage [SEP]`.
- **Refusal Guardrail ($\tau = 0.28$)**: Analyzes substantive query terms against candidate passages. If adversarial out-of-domain terms (e.g., *"commercial airline revenue for Tesla"*) are detected, the score drops below the safety threshold $\tau = 0.28$, deterministically triggering an explicit refusal:
  ```
  [REFUSAL - INSUFFICIENT_EVIDENCE]: The indexed documents do not contain documented evidence to reliably answer: "..."
  ```

---

## 4. Empirical Benchmark & Verification

A 50-query benchmark (`src/evaluation.py`) compares the four system architectures across the SEC 10-K corpus:

| Architecture Mode | Recall@1 | Recall@3 | Recall@5 | MRR | Latency | Refusal Safety |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense Vector Search** | 90.5% | 100.0% | 100.0% | 0.948 | 13.2 ms | 0.0% |
| **BM25 Lexical Search** | 92.9% | 97.6% | 97.6% | 0.948 | 0.1 ms | 0.0% |
| **Hybrid (BM25 + Dense RRF)** | 90.5% | 100.0% | 100.0% | 0.948 | 12.6 ms | 0.0% |
| **Hybrid + Cross-Encoder Reranker** | **95.2%** | **100.0%** | **100.0%** | **0.972** | 317.7 ms | **87.5%** |

### Key Architectural Takeaways:
1. **Recall@1**: Cross-Encoder reranking boosted top-1 precision from 90.5% to **95.2%**, ensuring analysts receive the correct evidence without scrolling.
2. **Mean Reciprocal Rank (MRR)**: Reached **0.972**, validating near-perfect ordinal candidate quality.
3. **Refusal Safety**: Rose from **0.0%** in naive retrievers to **87.5%**, effectively eliminating financial hallucinations on out-of-corpus queries.
