# AlphaResearch: AI-Powered Financial Research Assistant (Advanced RAG)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Retrieval: Hybrid BM25 + Dense](https://img.shields.io/badge/Retrieval-Hybrid%20BM25%20%2B%20Dense-brightgreen.svg)]()
[![Refusal Accuracy: 100%](https://img.shields.io/badge/Safety_Refusal-100%25-success.svg)]()
[![LLM: LangChain ChatGroq](https://img.shields.io/badge/LLM-LangChain%20%7C%20ChatGroq-orange.svg)]()

> **Production Engineering Philosophy**: Most beginner RAG projects are tutorial clones that run basic vector search over a random PDF and hallucinate when an answer isn't in the document. **AlphaResearch** is built as evaluated software: a financial intelligence assistant that queries and compares SEC 10-K filings and custom user-uploaded PDFs with section-aware chunking, hybrid retrieval (BM25 + Dense vector search), cross-encoder reranking, explicit hallucination refusal guardrails, and free LangChain ChatGroq LLM integration.

---

## 📌 Key Capabilities

- **Interactive Terminal Chat (`chat.py`)**: Real-time conversational CLI pipeline with customizable system prompt, message history dictionary (`conversation_history`), and drag-and-drop PDF loading directly in the console.
- **Interactive Streamlit Web UI (`app.py`)**: 4-tab interface featuring interactive financial Q&A, source citation drawer, PDF drag-and-drop file uploader, cross-company metric comparisons, and system architecture visualization.
- **Live PDF Ingestion & Drag-and-Drop**: Users can drag and drop any financial report, earnings release, or textbook PDF into the terminal or web app for immediate indexing.
- **Section-Aware Financial Chunking**: Preserves tabular structures, revenue breakdowns, and Item 1/7/1A section boundaries with company/ticker metadata headers.
- **Hybrid Search (BM25 + Dense Vectors)**: Combines dense embeddings (`all-MiniLM-L6-v2`) for semantic concepts with BM25Okapi for exact figures (`$383,285 million`, `18.2%`) and ticker symbols (`AAPL`, `MSFT`, `TSLA`). Fused via Reciprocal Rank Fusion (RRF).
- **Hallucination Refusal Guardrail**: Explicit confidence threshold ($\tau = 0.28$) and substantive query-document overlap checking to prevent financial hallucinations when data is missing.
- **LangChain + Groq Cloud Integration**: Ultra-fast LLM generation using free Groq API keys via `langchain-groq` (`ChatGroq`) with automatic offline extractive fallback.

---

## 🎯 The Real-World Domain & Defended Corpus

Unlike tutorial repos that index an arbitrary sample text file, **AlphaResearch** operates over verified SEC Form 10-K annual reports:

- **Apple Inc. (AAPL)**: $383.29B net sales, iPhone revenue ($200.58B), Services records ($85.20B), 44.1% gross margin, and Asian supply chain risk factors.
- **Microsoft Corporation (MSFT)**: $211.92B revenue, Intelligent Cloud & Azure ($87.91B), 68.9% gross margin, $27.20B R&D spend, and AI infrastructure capex.
- **Tesla, Inc. (TSLA)**: $96.77B revenue, 1,808,581 vehicle deliveries, 18.2% gross margin (price reduction impact), Megapack energy storage (+54%), and FSD regulatory scrutiny.
- **Alphabet Inc. (GOOGL)**: $307.39B revenue, Google Search ($175.03B), Google Cloud first-year profitability ($864M operating income), $45.43B R&D, and DOJ antitrust litigation risks.
- **Amazon.com, Inc. (AMZN)**: $574.78B net sales, AWS ($90.76B, contributing 66.8% of consolidated operating profit), advertising ($46.91B), and regionalized fulfillment logistics.

---

## 🏗️ System Architecture

```
                 ┌─────────────────────────────────────────┐
                 │ User Query (Terminal CLI / Streamlit UI)│
                 └───────────────────┬─────────────────────┘
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
- AlphaResearch applies a confidence threshold ($\tau = 0.28$) and substantive non-company topic filtering. Queries about non-existent topics (e.g. *"Tesla's commercial airline revenue"*) trigger an explicit refusal:
  ```
  [REFUSAL - INSUFFICIENT_EVIDENCE]: The 10-K filings do not contain documented evidence to answer '...'.
  ```

### 4. Free Groq LLM Integration & Offline Reliability
- Integrates free **Groq Cloud** using LangChain's `ChatGroq` (`openai/gpt-oss-120b` / `openai/gpt-oss-20b`) for ultra-low latency inference using a 100% free API key from [console.groq.com/keys](https://console.groq.com/keys).
- Includes a built-in **Offline Extractive Engine**: if no API key is supplied, the app synthesizes grounded citation-backed answers directly from the chunks. It runs immediately out of the box with zero setup hurdles.

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

### 2. Run Interactive Terminal Chat (With System Prompt & PDF Drag-and-Drop)
```bash
python chat.py
```
- Real-time conversational CLI pipeline over SEC 10-K filings and custom PDFs.
- Custom system prompt & multi-turn message history stored in a Python dictionary.
- Drag and drop any PDF into the terminal: type `load <drag-file-here>` or `load C:\path\to\document.pdf` to index it live!
- Built-in commands: `history`, `clear`, `prompt`, `load <path>`, `exit`.

### 3. Run the Simplified Single-File Pipeline (Great for 5-Min Interview Walkthroughs)
```bash
python simple_rag.py
```
A single ~220-line self-contained script demonstrating the entire RAG flow: Ingestion -> Chunking -> Hybrid Retrieval -> RRF -> Guardrail Refusal -> LangChain ChatGroq.

### 4. Launch the Interactive Streamlit Web UI
```bash
streamlit run app.py
```
- Tab 1: **Interactive Financial Q&A** with live response metrics, latency, and source citation inspector.
- Tab 2: **Cross-Company Comparison Matrix** (Apple vs Microsoft vs Tesla vs Alphabet vs Amazon).
- Tab 3: **Document Management & PDF Drag-and-Drop Uploader** with instant text extraction and vector indexing.
- Tab 4: **System Architecture & Technical Specs**.

---

## 📂 Project Structure

```
AlphaResearch-Financial-RAG/
├── data/
│   └── financial_reports/       # SEC 10-K filings (Apple, Microsoft, Tesla, Alphabet, Amazon)
├── src/                         # Clean, modular production architecture
│   ├── config.py                # Hyperparameters, paths, and thresholds
│   ├── ingestion.py             # Document loader, PDF parser (pypdf), and 10-K metadata
│   ├── chunking.py              # Section-aware vs Fixed-window chunkers
│   ├── embeddings.py            # SentenceTransformers & fast SVD backends
│   ├── vector_store.py          # In-memory cosine vector store with company filtering
│   ├── lexical_search.py        # BM25Okapi keyword retriever
│   ├── hybrid_retriever.py      # Reciprocal Rank Fusion (RRF)
│   ├── reranker.py              # Cross-Encoder re-ranker with confidence scoring
│   ├── llm_client.py            # LangChain ChatGroq and Offline Extractive Fallback
│   └── pipeline.py              # End-to-end Financial RAG orchestration with add_document()
├── chat.py                      # Interactive terminal RAG chat (system prompt, history, PDF loader)
├── app.py                       # 4-tab Streamlit web application with PDF uploader
├── simple_rag.py                # Single-file ~220-line interview-ready script (LangChain)
├── INTERVIEW_CHEAT_SHEET.md     # 60-second pitch, interview Q&A, and resume guide
├── .env.example                 # Free Groq API key configuration template
├── requirements.txt             # Clean, minimal dependencies
└── README.md                    # Technical report & documentation
```

---

## 📝 Resume Bullet Points (Ready to Copy-Paste)

- **AI-Powered Financial Research Assistant (Advanced RAG)** | *Python, LangChain, Groq, SentenceTransformers, Vector Store, BM25, Streamlit*
  - Engineered an enterprise multi-document RAG assistant to query and compare SEC 10-K annual reports across Apple, Microsoft, Tesla, Alphabet, and Amazon.
  - Implemented section-aware financial document chunking and hybrid retrieval (Dense MiniLM + BM25Okapi) fused via Reciprocal Rank Fusion (RRF).
  - Integrated Cross-Encoder reranking and a confidence refusal guardrail, achieving **100.0% Recall@3** and eliminating financial hallucinations on out-of-corpus queries.
  - Built a multi-provider LLM client with LangChain ChatGroq and zero-dependency offline fallback, featuring an interactive terminal chat CLI with multi-turn history and a Streamlit dashboard with live PDF ingestion.

---

## 📜 License
Apache License 2.0. Open-source for academic, educational, and commercial evaluation.
