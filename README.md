# AlphaResearch: Enterprise Financial Document Intelligence (Advanced RAG)

[![Live App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://namannn28-alpharesearch-financial-rag-app-ux5ypx.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Retrieval: Hybrid BM25 + Dense](https://img.shields.io/badge/Retrieval-Hybrid%20BM25%20%2B%20Dense-brightgreen.svg)]()
[![Refusal Safety: 87.5%](https://img.shields.io/badge/Refusal_Safety-87.5%25-success.svg)]()
[![LLM: LangChain ChatGroq](https://img.shields.io/badge/LLM-LangChain%20%7C%20ChatGroq-orange.svg)]()

> **AlphaResearch** is a production-style, evaluated Retrieval-Augmented Generation (RAG) platform engineered to query, analyze, and compare audited SEC 10-K annual filings (Apple, Microsoft, Tesla, Alphabet, Amazon) and custom corporate financial PDFs with zero hallucination.

---

## 🌐 Live Interactive Demo

🚀 **Experience the live deployment directly in your browser:**  
👉 **[https://namannn28-alpharesearch-financial-rag-app-ux5ypx.streamlit.app/](https://namannn28-alpharesearch-financial-rag-app-ux5ypx.streamlit.app/)**

| Service | Details |
| :--- | :--- |
| **Live App URL** | [namannn28-alpharesearch-financial-rag-app-ux5ypx.streamlit.app](https://namannn28-alpharesearch-financial-rag-app-ux5ypx.streamlit.app/) |
| **Hosting Platform** | Streamlit Community Cloud |
| **Inference Engine** | Hybrid Dense + BM25 Retrieval & RRF with Groq LLM Cloud Acceleration |
| **Corpus** | Audited SEC 10-Ks (AAPL, MSFT, TSLA, GOOGL, AMZN) + In-Memory Custom PDF Ingestion |

---

## 📌 Core Engineering Highlights

Unlike naive vector-search demos that hallucinate over unstructured text, AlphaResearch implements an enterprise-grade retrieval pipeline:

1. **Section-Aware Financial Chunking (`src/chunking.py`)**:
   - Preserves financial table structures, income statement rows, and MD&A headings without breaking numerical tables across arbitrary token boundaries.
   - Enriches every chunk with metadata headers: `[Company (Ticker) | Section Title | Page X]`.

2. **Dense + BM25 Hybrid Retrieval (`src/hybrid_retriever.py`)**:
   - Combines semantic generalization (`sentence-transformers/all-MiniLM-L6-v2`) with exact token matching (`BM25Okapi`) for financial metrics (`$383,285 million`, `44.1%`) and stock tickers (`AAPL`, `MSFT`, `TSLA`).
   - Fuses multi-channel candidate lists using **Reciprocal Rank Fusion (RRF)**.

3. **Cross-Encoder Reranking & Adversarial Refusal Guardrail (`src/reranker.py`)**:
   - Evaluates deep query-passage interactions using `cross-encoder/ms-marco-MiniLM-L-6-v2`.
   - Calibrated confidence threshold ($\tau = 0.28$) actively intercepts and rejects ungrounded or adversarial questions (e.g., *"What was Tesla's commercial airline ticket revenue?"*), returning structured evidence refusals.

4. **Multi-User Session Isolation & Privacy (`app.py`)**:
   - Clones pipeline vector stores in memory for each active browser session (`create_session_pipeline`).
   - Files uploaded by User A remain strictly isolated and invisible to User B.

5. **Multi-Turn Context Resolution (`src/pipeline.py`)**:
   - Inspects conversation history to resolve contextual follow-up inquiries (e.g., *"yes give me 2023 sales"*) to the previously referenced company.

6. **Dual Inference Engine with Zero-Crash Fallback (`src/llm_client.py`)**:
   - Cloud inference via **LangChain ChatGroq** (`llama-3.3-70b-versatile` / `llama-3.1-8b-instant`).
   - Automated offline deterministic extractive synthesis if API credentials or network connections are unavailable.

---

## 🔬 Empirical RAG Evaluation Benchmark

To quantitatively validate retrieval performance and hallucination prevention, AlphaResearch includes an automated benchmark suite (`src/evaluation.py`) over 50 evaluation items (42 factual questions across audited 10-K filings + 8 out-of-corpus adversarial hallucination traps):

```bash
python src/evaluation.py
```

### Benchmark Results (50 Test Queries)

| Architecture Mode | Recall@1 | Recall@3 | Recall@5 | MRR | Latency (ms) | Refusal Safety |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense Vector Search** | 90.5% | 100.0% | 100.0% | 0.948 | 13.2 ms | 0.0% |
| **BM25 Lexical Search** | 92.9% | 97.6% | 97.6% | 0.948 | 0.1 ms | 0.0% |
| **Hybrid (BM25 + Dense RRF)** | 90.5% | 100.0% | 100.0% | 0.948 | 12.6 ms | 0.0% |
| **Hybrid + Cross-Encoder Reranker** | **95.2%** | **100.0%** | **100.0%** | **0.972** | 317.7 ms | **87.5%** |

### Key Metrics Defined:
- **Recall@K**: Measures the proportion of queries where the ground-truth answer chunk appeared within the top-K retrieved candidates.
- **MRR (Mean Reciprocal Rank)**: Calculates the average reciprocal rank ($\frac{1}{\text{rank}}$) of the first relevant chunk, rewarding higher placement.
- **Refusal Safety**: Measures the system's ability to reject adversarial queries instead of hallucinating plausibly worded approximations.

---

## 🏗️ System Architecture

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

For detailed architectural specifications, formulas, and component descriptions, see [docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md).

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- Git

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/Namannn28/AlphaResearch-Financial-RAG.git
cd AlphaResearch-Financial-RAG

# Create and activate virtual environment
# Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS:
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. API Key Setup (Optional)
Cloud generation utilizes the free Groq Cloud API. Copy `.env.example` to `.env` or set your key:
```bash
# In .env (or export in shell)
GROQ_API_KEY="your-groq-api-key-here"
```
*(Note: If no API key is provided, the pipeline automatically operates using its built-in offline extractive fallback engine).*

### 4. Running the Application

#### Streamlit Web Interface:
```bash
streamlit run app.py
```
- Direct single-page chat stream.
- In-memory custom PDF drag-and-drop uploader.
- Instant page-level source citations and latency telemetry.

#### Terminal Conversational CLI:
```bash
python chat.py
```
- Interactive multi-turn CLI with conversation memory.
- Commands: `history`, `clear`, `load <file.pdf>`, `exit`.

---

## ☁️ Deployment to Streamlit Community Cloud

1. Fork or push this repository to your GitHub account: `https://github.com/Namannn28/AlphaResearch-Financial-RAG`.
2. Navigate to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. Click **"New app"** and configure:
   - **Repository**: `Namannn28/AlphaResearch-Financial-RAG`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Expand **"Advanced settings"** -> **"Secrets"** and enter:
   ```toml
   GROQ_API_KEY = "your-actual-groq-api-key"
   ```
5. Click **"Deploy"**. Streamlit Community Cloud will automatically install dependencies from `requirements.txt` and launch your live application.

---

## 📂 Repository Structure

```
AlphaResearch-Financial-RAG/
├── data/
│   ├── evaluation/              # 50-item benchmark dataset & evaluation results
│   │   ├── financial_eval_dataset.json
│   │   └── financial_benchmark_results.json
│   └── financial_reports/       # Audited SEC 10-K filings (AAPL, MSFT, TSLA, GOOGL, AMZN)
├── docs/
│   └── SYSTEM_DESIGN.md         # Comprehensive system architecture & mathematical specifications
├── src/                         # Modular pipeline components
│   ├── config.py                # Hyperparameters, paths, and thresholds
│   ├── query_preprocessor.py    # Intent detection, greeting handling, and spelling correction
│   ├── ingestion.py             # Multi-format document loader and PDF parser (pypdf)
│   ├── chunking.py              # Section-aware chunker preserving tables and page markers
│   ├── embeddings.py            # SentenceTransformers dense vector embeddings (all-MiniLM-L6-v2)
│   ├── vector_store.py          # In-memory cosine vector store with company filtering
│   ├── lexical_search.py        # BM25Okapi exact keyword & metric retriever
│   ├── hybrid_retriever.py      # Reciprocal Rank Fusion (RRF) combining Dense + BM25
│   ├── reranker.py              # Cross-Encoder re-ranker with confidence refusal guardrail
│   ├── llm_client.py            # LangChain ChatGroq client with citation formatting
│   ├── evaluation.py            # Automated evaluation harness (Recall@K, MRR, safety)
│   └── pipeline.py              # End-to-end orchestration pipeline
├── app.py                       # Single-page Streamlit web application
├── chat.py                      # Interactive terminal RAG chat with multi-turn memory
├── requirements.txt             # Clean dependency specifications
└── README.md                    # Project documentation
```

---

## 📜 License
Distributed under the Apache 2.0 License. See `LICENSE` for more information.
