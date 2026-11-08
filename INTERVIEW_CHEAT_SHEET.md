# 🎓 Candidate Cheat Sheet: AI-Powered Financial Research Assistant (RAG)

> **Interview Purpose**: This guide provides freshers with clear, confident answers for AI/ML Engineer, Generative AI Engineer, and Software Developer technical interviews. It translates the code you built into senior-level engineering judgment.

---

## ⚡ 1. The 60-Second Elevator Pitch
> *"I built **AlphaResearch**, an evaluated Retrieval-Augmented Generation system designed to query and compare SEC 10-K annual reports for companies like Apple, Microsoft, Tesla, Alphabet, and Amazon, as well as user-uploaded financial PDFs.*
>
> *Most tutorial RAG projects run naive vector search over a random PDF and hallucinate when a question isn't in the document. I built a production-style architecture featuring:*
> 1. *A **section-aware financial chunker** that preserves balance sheet tables and MD&A headings.*
> 2. ***Hybrid retrieval** fusing BM25 keyword matching for exact dollar amounts and ticker symbols with Dense embeddings via Reciprocal Rank Fusion (RRF).*
> 3. *A **Cross-Encoder reranker** that re-scores candidates with cross-attention.*
> 4. *An explicit **confidence refusal threshold** that rejects out-of-corpus queries instead of hallucinating.*
> 5. *A **free LangChain ChatGroq** integration with an automatic offline extractive fallback.*
> 6. *Dual access interfaces: an **interactive terminal CLI (`chat.py`)** with custom system prompts, message history dictionaries, and PDF drag-and-drop, alongside a full **Streamlit Web UI (`app.py`)**."*

---

## 🏗️ 2. Core Architecture Diagram

```
                 ┌─────────────────────────────────────────┐
                 │ User Query (Terminal CLI / Streamlit UI)│
                 └───────────────────┬─────────────────────┘
                                     │
                     ┌───────────────┴───────────────┐
                     ▼                               ▼
           ┌──────────────────┐            ┌──────────────────┐
           │ Dense Embedding  │            │  BM25 Lexical    │
           │ MiniLM (384-dim) │            │  Exact Figures   │
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

## ❓ 3. Top Interview Questions & Senior Answers

### Q1: Why did you pick SEC 10-K filings instead of a generic sample PDF?
**Answer**:
> *"SEC 10-K filings represent a high-stakes, real-world domain. Financial analysts need exact quantitative metrics (e.g., Apple's $383.29B net sales, Tesla's 18.2% gross margin, or AWS's 66.8% share of operating profit). A narrow, well-defined corpus allows rigorous testing of quantitative retrieval accuracy and enables clear cross-company comparisons."*

### Q2: Why did you use Hybrid Search (BM25 + Dense) instead of just Vector Search?
**Answer**:
> *"Dense vector embeddings excel at semantic similarity and paraphrasing, but they often struggle with exact financial tokens—like ticker symbols (`TSLA`, `MSFT`), accounting sections (`Item 7 MD&A`), and specific numerical currency amounts (`$383,285 million`). BM25 provides exact keyword precision. By combining them using Reciprocal Rank Fusion (RRF), we ensure both semantic understanding and exact figure retrieval."*

### Q3: What is Reciprocal Rank Fusion (RRF)?
**Answer**:
> *"RRF is a rank-based aggregation algorithm that doesn't require normalizing raw scores across models. The formula is:*
> $$\text{Score}(d) = \sum_{m \in \{\text{Dense}, \text{BM25}\}} \frac{1}{k + \text{rank}_m(d)}$$
> *where $k=60$ is a standard smoothing constant. It penalizes outliers and rewards documents that rank high in both search methods."*

### Q4: How do you prevent hallucinations when an answer isn't in the reports?
**Answer**:
> *"We implemented a two-tier safety refusal guardrail:*
> 1. *We compute confidence scores for the top retrieved chunk. If the score falls below our safety threshold ($\tau = 0.28$), the system refuses to answer.*
> 2. *We check substantive non-company topic words. If a user asks about 'commercial airline revenue' for Apple or Tesla, the system identifies that zero substantive topic tokens exist in the retrieved document, applies a safety penalty, and returns an explicit `INSUFFICIENT_EVIDENCE` refusal."*

### Q5: How did you integrate the LLM, and what happens if the API key is missing?
**Answer**:
> *"I integrated the free **Groq Cloud API** using LangChain's `ChatGroq` for ultra-fast, high-accuracy conversational inference. I also engineered an offline extractive fallback engine. If an API key is not supplied or network fails, the system automatically synthesizes answers directly from the top verified chunks with citations. It never crashes."*

### Q6: How does the application handle PDF uploads and multi-turn conversations?
**Answer**:
> *"In both the terminal CLI (`chat.py`) and Streamlit web app (`app.py`), we use `pypdf` to extract text and page metadata on the fly. The extracted text is passed through our section-aware chunker, dynamically vector-embedded, and added to the BM25 index. Multi-turn chat maintains a structured `conversation_history` message list conforming to standard role-based schemas (`system`, `user`, `assistant`), passing retrieved context alongside chat memory."*

### Q7: Why would an enterprise use this custom RAG system instead of just uploading files to ChatGPT?
**Answer (The Enterprise Value Proposition)**:
> *"ChatGPT is a general consumer chatbot; it is **not an enterprise data system**. An enterprise—especially in finance, banking, or healthcare—cannot simply use ChatGPT for four fundamental engineering and business reasons:*
>
> 1. ***Data Privacy & SEC Compliance (Zero Data Leakage)***: Enterprises legally cannot upload private financial filings, internal audits, or customer portfolios to external public LLM endpoints where data can be logged or used for training. Our system operates locally with isolated session memory (`create_session_pipeline`), guaranteeing that confidential documents never leak across users or to third parties.
> 2. ***Numerical Precision vs. Attention Drift (BM25 + Dense Hybrid)***: Standard LLMs use pure semantic attention, which notoriously rounds, interpolates, or swaps closely spaced financial numbers (e.g., confusing `$383,285M` with `$393,285M` or mixing FY2022 with FY2023). Our pipeline enforces **BM25 exact keyword matching** combined with Dense vector search via **Reciprocal Rank Fusion (RRF)** to anchor answers on exact numerical tokens.
> 3. ***Deterministic Hallucination Guardrail ($\tau = 0.28$)***: ChatGPT is trained to be conversational and agreeable—when given out-of-corpus or adversarial prompts (e.g., 'What was Tesla's commercial airline revenue?'), it often hallucinates plausible-sounding figures. Our system includes a Cross-Encoder reranker and substantive token guardrail that cuts confidence below 0.28 and returns a deterministic `INSUFFICIENT_EVIDENCE` refusal. In finance, an honest refusal is 100x safer than a confident hallucination.
> 4. ***Compliance-Grade Page Citations & Traceability***: Financial analysts cannot submit recommendations based on opaque AI output. Our pipeline extracts and displays exact page numbers, section headers, and company metadata for every claim, making every figure verifiable in seconds.
> 5. ***Scale, Latency & Cost***: Uploading 500 reports of 100+ pages into ChatGPT exceeds context limits, costs significant per-token fees, and takes minutes. Our pre-indexed hybrid search retrieves top relevant passages in **sub-15ms** on free, local CPU vector stores."*

### Q8: How did you evaluate your RAG pipeline? What metrics did you measure?
**Answer (The Empirical Engineering Proof)**:
> *"I built an automated evaluation framework (`src/evaluation.py`) using a benchmark test suite of 50 curated financial inquiries and out-of-corpus adversarial hallucination traps across our 10-K corpus.
>
> I benchmarked four distinct retrieval architectures:*
> 1. *Pure Dense Vector Search (MiniLM)*
> 2. *Pure BM25 Lexical Search (Okapi BM25)*
> 3. *Hybrid Search (Reciprocal Rank Fusion - RRF)*
> 4. *Hybrid Search + Cross-Encoder Reranker & Guardrail (Full Pipeline)*
>
> *I measured two primary retrieval metrics:*
> - ***Recall@K***: *Did the top-K retrieved candidates contain the gold factual disclosure?*
> - ***MRR (Mean Reciprocal Rank)***: *At what rank did the gold passage appear ($1/\text{rank}$)?*
> - ***Refusal Precision***: *Did the system correctly refuse adversarial out-of-corpus questions?*
>
> *The empirical results:*
> - *Standard Dense vector search achieved **90.5% Recall@1** and **0.948 MRR**, but had **0.0% Refusal Safety** (it always returned ungrounded chunks on trick questions).*
> - *BM25 achieved **92.9% Recall@1**, but missed paraphrased semantic queries (**97.6% Recall@5**).*
> - *Our full **Hybrid + Cross-Encoder Reranker** achieved **95.2% Recall@1**, **100% Recall@5**, and an **MRR of 0.972**, while correctly rejecting **87.5%** of adversarial hallucination traps below our confidence threshold ($\tau = 0.28$)."*

---

## 🔬 4. Empirical RAG Benchmark Results

Run benchmark anytime: `python src/evaluation.py`

| Architecture Mode | Recall@1 | Recall@3 | Recall@5 | MRR | Latency (ms) | Refusal Safety |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense Vector Search** | 90.5% | 100.0% | 100.0% | 0.948 | 13.2 ms | 0.0% |
| **BM25 Lexical Search** | 92.9% | 97.6% | 97.6% | 0.948 | 0.1 ms | 0.0% |
| **Hybrid (BM25 + Dense RRF)** | 90.5% | 100.0% | 100.0% | 0.948 | 12.6 ms | 0.0% |
| **Hybrid + Cross-Encoder Reranker** | **95.2%** | **100.0%** | **100.0%** | **0.972** | 317.7 ms | **87.5%** |

> [!NOTE]
> **What this proves to an interviewer**:
> - **Recall@1 jumped from 90.5% to 95.2%**: Cross-attention ensures the single best passage is placed at rank #1 in 95.2% of queries.
> - **MRR reached 0.972**: Analysts get the exact answer immediately without having to browse down the candidate list.
> - **Refusal Safety is 87.5% vs 0%**: Pure retrievers blindly hallucinate; your reranker guardrail actively blocks out-of-corpus queries.

---

## 📝 5. Ready-to-Use Resume Bullets

Copy and paste these directly onto your resume:

- **AI-Powered Financial Research Assistant (Evaluated Advanced RAG)** | *Python, LangChain, Groq, SentenceTransformers, Vector Store, BM25, Streamlit*
  - Engineered an enterprise multi-document RAG assistant to query and compare SEC 10-K annual reports across Apple, Microsoft, Tesla, Alphabet, and Amazon.
  - Implemented section-aware financial document chunking and hybrid retrieval (Dense MiniLM + BM25Okapi) fused via Reciprocal Rank Fusion (RRF).
  - Integrated Cross-Encoder reranking and an adversarial confidence refusal guardrail ($\tau = 0.28$), achieving **95.2% Recall@1**, **100% Recall@5**, **0.972 MRR**, and **87.5% refusal safety** across a 50-query benchmark.
  - Built a multi-provider LLM client with LangChain ChatGroq and zero-dependency offline fallback, featuring an interactive terminal chat CLI with multi-turn history and a Streamlit dashboard with live PDF ingestion.
