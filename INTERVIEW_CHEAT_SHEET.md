# 🎯 CandidateFit-RAG: The Fresher's Interview Cheat-Sheet
*How to explain every concept in plain English without getting nervous or tangled up in jargon.*

---

## 1. The Core 60-Second Elevator Pitch
When the interviewer asks: **"Tell me about this RAG project on your resume."**

> **What to say:**  
> *"Most introductory RAG tutorials build a basic 'chat with PDF' bot that slices text blindly every 200 words and hallucinates whenever an answer isn't in the document.  
> I wanted to build something that solves a real hiring problem: auditing candidate resumes against job descriptions.  
> I focused on three practical improvements:  
> 1. **Section-Aware Chunking**: Resumes have natural sections (Experience, Projects, Skills). Keeping an entire job role or project in one chunk ensures metrics like '$42k saved' stay attached to the tools used (Redis + Milvus).  
> 2. **Hybrid Search**: Dense vectors can confuse acronyms like 'vLLM', 'Triton', and 'PyTorch'. I combined BM25 keyword matching with dense MiniLM embeddings via Reciprocal Rank Fusion so exact acronyms are never missed.  
> 3. **The 'I Don't Know' Guardrail**: If an interviewer asks if a candidate has COBOL or Rust experience and it's not in the resume, my pipeline explicitly refuses with `INSUFFICIENT_EVIDENCE` instead of hallucinating.  
> I wrote a 37-question evaluation test set and measured a 100% Recall@3 and a +32% precision lift with cross-encoder re-ranking."*

---

## 2. Common Interview Questions & Simple Answers

### Q1: "Why didn't you just use OpenAI embeddings?"
> **Answer:**  
> *"Two reasons: **Cost/Privacy** and **RAM efficiency**.  
> In hiring, candidate resumes contain Personally Identifiable Information (PII) like phone numbers and previous employers, which enterprise HR teams often can't send to external third-party cloud APIs.  
> By using `all-MiniLM-L6-v2`, it runs 100% locally on CPU in under 15ms with zero token cost. Plus, MiniLM outputs 384 dimensions compared to OpenAI's 1536, saving 75% RAM on vector storage."*

### Q2: "What is Reciprocal Rank Fusion (RRF) and why did you use it?"
> **Answer:**  
> *"When you combine vector similarity and BM25 keywords, their score scales are completely different: cosine similarity is between -1 and +1, while BM25 scores can be any positive number like 14.8.  
> If you try to add them directly, BM25 dominates.  
> Reciprocal Rank Fusion (RRF) solves this simply: instead of adding raw scores, it adds the reciprocal of their ranks:  
> `RRF Score = 1 / (60 + rank_dense) + 1 / (60 + rank_bm25)`  
> It's scale-invariant and treats both searchers fairly without needing complex score calibration."*

### Q3: "What is the difference between your Bi-Encoder and Cross-Encoder?"
> **Answer:**  
> *"A Bi-Encoder (MiniLM) converts the question and candidate chunks into vectors separately, which is ultra-fast for filtering down 1,000s of chunks. But because it encodes them separately, it misses subtle interactions.  
> A Cross-Encoder passes the question AND the chunk together through all transformer attention layers simultaneously. It's much more accurate, but slower.  
> That's why I used a two-stage approach: the Bi-Encoder quickly retrieves the top-6 candidate chunks, and the Cross-Encoder re-ranks those top-6 down to the top-3. This gave us a +32% boost in top-1 precision."*

### Q4: "How does your system prevent hallucinations?"
> **Answer:**  
> *"We set a confidence threshold ($\tau = 0.32$) on the top retrieved chunk.  
> If an applicant never used Rust or COBOL, the keyword and semantic overlap is near zero. When the pipeline sees a score below 0.32, it bypasses answer generation and returns:  
> `INSUFFICIENT_EVIDENCE: The resume does not contain documented evidence regarding [topic].`  
> In testing, this caught 100% of our adversarial trap questions."*

### Q5: "What would you improve if you had more time?"
> **Answer:**  
> *"Three things:  
> 1. Multi-hop synthesis across multiple candidates at once (e.g. 'Compare Candidate A and B across AWS spend reduction').  
> 2. OCR preprocessing for graphical or two-column PDF resumes before markdown conversion.  
> 3. Quantizing the cross-encoder to ONNX/INT8 to cut re-ranking latency down to under 5ms."*

---

## 3. Code Walkthrough Quick-Reference

If an interviewer asks you to show the code:
1. Open [`rag_pipeline.py`](file:///c:/Users/HP/Desktop/rag%20project/rag_pipeline.py):  
   - Show `chunk_resume_by_sections()` (Line 50): explains how markdown headers preserve context.  
   - Show `SimpleBM25Searcher` (Line 100): explains how keywords complement vectors.  
   - Show `SimpleCandidateRAG.query()` (Line 180): shows the retrieval, confidence check, and refusal message.
2. Open [`evaluation/run_eval.py`](file:///c:/Users/HP/Desktop/rag%20project/evaluation/run_eval.py):  
   - Show how you benchmarked chunking strategies and measured Hit@K and MRR.
3. Open [`app.py`](file:///c:/Users/HP/Desktop/rag%20project/app.py):  
   - Show the Streamlit UI with the interactive candidate audit and the 112 Job Descriptions!
