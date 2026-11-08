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

---

## 📝 4. Ready-to-Use Resume Bullets

Copy and paste these directly onto your resume:

- **AI-Powered Financial Research Assistant (Advanced RAG)** | *Python, LangChain, Groq, SentenceTransformers, Vector Store, BM25, Streamlit*
  - Engineered an enterprise multi-document RAG assistant to query and compare SEC 10-K annual reports across Apple, Microsoft, Tesla, Alphabet, and Amazon.
  - Implemented section-aware financial document chunking and hybrid retrieval (Dense MiniLM + BM25Okapi) fused via Reciprocal Rank Fusion (RRF).
  - Integrated Cross-Encoder reranking and a confidence refusal guardrail, achieving high retrieval precision and eliminating financial hallucinations on out-of-corpus queries.
  - Built a multi-provider LLM client with LangChain ChatGroq and zero-dependency offline fallback, featuring an interactive terminal chat CLI with multi-turn history and a Streamlit dashboard with live PDF ingestion.
