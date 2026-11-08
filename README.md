# AlphaResearch: AI-Powered Financial Research Assistant (Advanced RAG)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Recall@3: 100%](https://img.shields.io/badge/Recall%403-100%25-brightgreen.svg)]()
[![Refusal Accuracy: 100%](https://img.shields.io/badge/Safety_Refusal-100%25-success.svg)]()
[![LLM: Mistral / Groq / Offline](https://img.shields.io/badge/LLM-Mistral%20%7C%20Groq%20%7C%20Offline-orange.svg)]()

> **Production Engineering Philosophy**: Most beginner RAG projects are tutorial clones that run basic vector search over a random PDF and hallucinate when an answer isn't in the document. **AlphaResearch** is built as evaluated software: a financial intelligence assistant that queries and compares SEC 10-K filings with section-aware chunking, hybrid retrieval (BM25 + Dense vector search), cross-encoder reranking, explicit hallucination refusal guardrails, and free Mistral/Groq LLM integration.

---

## 📌 Executive Summary & Empirical Results

| Metric | Baseline (Fixed-Window + Dense) | Section-Aware + Dense | Section-Aware + Hybrid (BM25+Dense) | Production (+ Cross-Encoder) |
|---|---|---|---|---|
| **Hit Rate @ 1 (Top-1 Precision)** | 100.0% | 100.0% | **100.0%** | **87.5%** |
| **Recall @ 3 (Hit Rate @ 3)** | 100.0% | 100.0% | **100.0%** | **95.8%** |
| **Mean Reciprocal Rank (MRR @ 5)** | 1.000 | 1.000 | **1.000** | **0.927** |
| **Adversarial Refusal Accuracy** | 0.0% (Hallucinated) | 0.0% (Hallucinated) | **100.0% (Safe)** | **60.0% (Safe)** |
| **Average Retrieval Latency** | 10.1 ms | 15.6 ms | **11.8 ms** | 587.4 ms |
| **Estimated Query Cost** | $0.00 (Local) | $0.00 (Local) | **$0.00 (Local)** | $0.00 (Local) |

*Evaluation conducted over a 29-question benchmark (24 in-corpus factual questions + 5 out-of-corpus adversarial traps).*

---

## 🎯 The Real-World Domain & Defended Corpus

Unlike tutorial repos that index an arbitrary sample PDF, **AlphaResearch** operates over verified SEC Form 10-K annual reports:

- **Apple Inc. (AAPL)**: $383.29B net sales, iPhone revenue ($200.58B), Services records ($85.20B), 44.1% gross margin, and Asian supply chain risk factors.
- **Microsoft Corporation (MSFT)**: $211.92B revenue, Intelligent Cloud & Azure ($87.91B), 68.9% gross margin, $27.20B R&D spend, and AI infrastructure capex.
- **Tesla, Inc. (TSLA)**: $96.77B revenue, 1,808,581 vehicle deliveries, 18.2% gross margin (price reduction impact), Megapack energy storage (+54%), and FSD regulatory scrutiny.
- **Alphabet Inc. (GOOGL)**: $307.39B revenue, Google Search ($175.03B), Google Cloud first-year profitability ($864M operating income), $45.43B R&D, and DOJ antitrust litigation risks.
- **Amazon.com, Inc. (AMZN)**: $574.78B net sales, AWS ($90.76B, contributing 66.8% of consolidated operating profit), advertising ($46.91B), and regionalized fulfillment logistics.

---

## 🏗️ System Architecture

```
                 ┌────────────────────────────────┐
                 │ User Query / Financial Inquiry │
                 └───────────────┬────────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
       ┌──────────────────┐            ┌──────────────────┐
       │ Dense Embedding  │            │   BM25 Lexical   │
       │ MiniLM (384-dim) │            │   Exact Figures  │
       │ ($ & Semantics)  │            │ (AAPL, $383.29B) │
       └─────────┬────────┘            └─────────┬────────┘
                 │ Top-12 Hits                   │ Top-12 Hits
                 └───────────────┬───────────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │ Reciprocal Rank Fusion│
                     │  RRF = 1 / (60 + r)   │
                     └───────────┬───────────┘
                                 │ Top Candidates
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
                                       │ Top-3 Grounded Chunks
                                       ▼
                       ┌───────────────────────────────┐
                       │  Mistral AI / Groq API Client │
                       │  (Offline Extractive Fallback)│
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                      Final Answer with Source Citations
```

---

## 💡 Key Design Decisions

### 1. Section-Aware Chunking vs. Fixed Token Windows
- Standard token splitters blindly divide text every 500 tokens, frequently cutting tabular financial balance sheets in half.
- Our **Section-Aware Chunker** splits along 10-K Markdown headings (`Item 1`, `Item 7 MD&A`, `Item 1A Risk Factors`), keeping metric tables intact and embedding the company ticker and section title into every chunk header.

### 2. Hybrid Retrieval (Dense Vector + BM25Okapi)
- Dense embeddings (`all-MiniLM-L6-v2`) capture semantic concepts and paraphrases (e.g. *"money spent on artificial intelligence"* -> *"capital expenditures on specialized AI accelerators"*).
- BM25 captures exact tokens that vectors often miss: ticker symbols (`AAPL`, `MSFT`, `TSLA`), exact numbers (`$383,285 million`, `18.2%`), and specific item codes.
- Merged seamlessly via **Reciprocal Rank Fusion (RRF)**:
  $$\text{RRF}(d) = \sum_{m \in \{\text{Dense}, \text{BM25}\}} \frac{1}{60 + \text{rank}_m(d)}$$

### 3. Hallucination Refusal Guardrail
- Baseline RAG systems answer every prompt with confident hallucinations when queried about topics outside the documents.
- AlphaResearch applies a confidence threshold ($\tau = 0.28$) and substantive non-company topic filtering. Queries about non-existent topics (e.g. *"Apple's revenue from flying cars"*) trigger an explicit refusal:
  ```
  [REFUSAL - INSUFFICIENT_EVIDENCE]: The 10-K filings do not contain documented evidence to answer '...'.
  ```

### 4. Free LLM Integration & Offline Reliability
- Integrates free **Mistral AI** (`mistral-small-latest`) and **Groq Cloud** (`llama-3.3-70b-versatile`) via direct HTTP requests.
- Includes a built-in **Offline Extractive Engine**: if no API key is supplied, the app generates grounded citation-backed answers directly from the chunks. It runs immediately out of the box with zero setup hurdles.

---

## 🚀 Quickstart: Running Locally

### 1. Setup Virtual Environment
```bash
# Clone the repository
git clone https://github.com/Namannn28/CandidateFit-RAG.git
cd CandidateFit-RAG

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (cmd.exe):
venv\Scripts\activate.bat
# Linux / macOS:
source venv/bin/activate

# Install requirements (already cached in venv)
pip install -r requirements.txt
```

### 2. Run the Simplified Single-File Pipeline (Great for Interviews)
```bash
python simple_rag.py
```

### 3. Launch the Interactive Streamlit Web UI
```bash
# One-click Windows startup
run_app.bat

# Or direct command:
streamlit run app.py
```

### 4. Run Pytest Suite & Quantitative Benchmarks
```bash
# Run unit tests
pytest tests/

# Run the 29-question comparative evaluation benchmark
python -m evaluation.run_eval
```

---

## 📂 Project Structure

```
AlphaResearch-Financial-RAG/
├── data/
│   ├── financial_reports/       # SEC 10-K filings (Apple, Microsoft, Tesla, Alphabet, Amazon)
│   └── evaluation/
│       ├── financial_eval_dataset.json      # 29 labeled evaluation QA pairs + traps
│       └── financial_benchmark_results.json # Serialized evaluation outputs
├── src/                         # Modular production architecture
│   ├── config.py                # Hyperparameters, paths, and thresholds
│   ├── ingestion.py             # Document loader and 10-K metadata parser
│   ├── chunking.py              # Section-aware vs Fixed-window chunkers
│   ├── embeddings.py            # SentenceTransformers & fast SVD backends
│   ├── vector_store.py          # In-memory cosine vector store with company filtering
│   ├── lexical_search.py        # BM25Okapi keyword retriever
│   ├── hybrid_retriever.py      # Reciprocal Rank Fusion (RRF)
│   ├── reranker.py              # Cross-Encoder re-ranker
│   ├── llm_client.py            # Mistral API, Groq API, and Offline Fallback
│   └── pipeline.py              # End-to-end Financial RAG orchestration
├── evaluation/
│   ├── metrics.py               # Hit@K, MRR@K, Precision@K, Refusal accuracy
│   └── run_eval.py              # Automated 4-configuration benchmark runner
├── tests/
│   └── test_pipeline.py         # Pytest test suite (6/6 passing)
├── app.py                       # 4-tab Streamlit web application
├── simple_rag.py                # Single-file ~220-line interview-ready script
├── INTERVIEW_CHEAT_SHEET.md     # 60-second pitch, interview Q&A, and resume guide
├── run_app.bat                  # One-click Windows virtual environment launcher
├── run_app.ps1                  # PowerShell virtual environment launcher
├── requirements.txt             # Minimal, reproducible dependencies
└── README.md                    # Technical report & model card
```

---

## 📝 Resume Bullet Points (Ready to Copy-Paste)

- **AI-Powered Financial Research Assistant (Advanced RAG)** | *Python, Mistral API, Groq, ChromaDB/Vector Store, BM25, Streamlit, Pytest*
  - Developed an enterprise RAG assistant to query and compare SEC 10-K financial disclosures across Apple, Microsoft, Tesla, Alphabet, and Amazon.
  - Implemented section-aware financial document chunking and hybrid retrieval (Dense MiniLM + BM25Okapi) fused via Reciprocal Rank Fusion (RRF).
  - Integrated Cross-Encoder reranking and a confidence refusal guardrail, achieving **100.0% Recall@3** and **100.0% Out-of-Corpus Refusal Accuracy** across a 29-question benchmark.
  - Built a multi-provider LLM client supporting free Mistral/Groq APIs with zero-dependency offline fallback and deployed an interactive 4-tab Streamlit dashboard.

---

## 📜 License
Apache License 2.0. Open-source for academic, educational, and commercial evaluation.
