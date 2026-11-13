"""
==============================================================================
STEP 2: SECTION-AWARE CHUNKING MODULE (src/chunking.py)
==============================================================================
Role in RAG Pipeline:
    Splits ingested documents into coherent, context-rich chunks while strictly
    preserving section titles, page markers, and financial tables. Prepend
    metadata headers to each chunk so embeddings retain company context.

Key Architectural & Design Principles:
    - Fixed-window chunking flaws: Blindly splitting every 500 tokens often
      slices financial tables in half, severing rows from column headers.
    - Section-Aware chunking: Splits on logical heading boundaries (Item 1,
      MD&A, Risk Factors) and paragraphs, keeping numerical disclosures intact.
    - Context enrichment: Every chunk begins with:
      `[Company (Ticker) | Section Title | Page X]`
      This prevents orphan chunks where numbers appear without their company.

Run directly to test:
    python src/chunking.py
==============================================================================
"""

import sys
from pathlib import Path

# Add project root to path for direct module execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import dataclass
from typing import List, Dict, Any
from src.ingestion import FinancialDocument, DocumentIngestionEngine
from src.config import REPORTS_DIR


@dataclass
class FinancialChunk:
    """A self-contained chunk with rich metadata for retrieval and citations."""
    chunk_id: str
    doc_id: str
    company: str
    ticker: str
    fiscal_year: str
    section_title: str
    text: str
    token_count_approx: int
    page_number: int = 1


class SectionAwareFinancialChunker:
    """
    Chunks financial documents respecting section and page boundaries.
    Ensures financial metrics and tables remain tied to their company and section.
    """

    def __init__(self, target_chunk_words: int = 250, min_chars: int = 25):
        self.target_chunk_words = target_chunk_words
        self.min_chars = min_chars

    def chunk_document(self, document: FinancialDocument) -> List[FinancialChunk]:
        """Splits a single document into section-aware chunks."""
        chunks: List[FinancialChunk] = []
        chunk_idx = 0

        for sec in document.sections:
            title = sec["title"]
            content = sec["content"]
            page = sec.get("page", 1)

            if len(content.strip()) < self.min_chars:
                continue

            # First split on double newlines if present; otherwise group by lines
            if "\n\n" in content:
                raw_paras = [p.strip() for p in content.split("\n\n") if p.strip()]
            else:
                # Group lines into ~30-50 word units so we don't have one gigantic paragraph
                lines = [l.strip() for l in content.split("\n") if l.strip()]
                raw_paras = []
                cur_p = []
                cur_wc = 0
                for line in lines:
                    cur_p.append(line)
                    cur_wc += len(line.split())
                    if cur_wc >= 40 or line.endswith((".", ":", ";")):
                        raw_paras.append(" ".join(cur_p))
                        cur_p = []
                        cur_wc = 0
                if cur_p:
                    raw_paras.append(" ".join(cur_p))

            # Break any overly large paragraphs (> target_chunk_words) into smaller pieces
            paragraphs: List[str] = []
            for p in raw_paras:
                words = p.split()
                if len(words) <= self.target_chunk_words:
                    paragraphs.append(p)
                else:
                    step = max(50, self.target_chunk_words - 40)
                    for i in range(0, len(words), step):
                        sub_p = " ".join(words[i:i + self.target_chunk_words])
                        if sub_p.strip():
                            paragraphs.append(sub_p)

            current_buffer: List[str] = []
            current_word_count = 0

            for para in paragraphs:
                para_words = len(para.split())
                if current_word_count + para_words > self.target_chunk_words and current_buffer:
                    chunk_text = "\n\n".join(current_buffer)
                    chunks.append(FinancialChunk(
                        chunk_id=f"{document.doc_id}_sec_{chunk_idx}",
                        doc_id=document.doc_id,
                        company=document.company,
                        ticker=document.ticker,
                        fiscal_year=document.fiscal_year,
                        section_title=title,
                        text=f"[{document.company} ({document.ticker}) | {title} | Page {page}]\n{chunk_text}",
                        token_count_approx=int(len(chunk_text.split()) * 1.3),
                        page_number=page
                    ))
                    chunk_idx += 1
                    current_buffer = [para]
                    current_word_count = para_words
                else:
                    current_buffer.append(para)
                    current_word_count += para_words

            if current_buffer:
                chunk_text = "\n\n".join(current_buffer)
                chunks.append(FinancialChunk(
                    chunk_id=f"{document.doc_id}_sec_{chunk_idx}",
                    doc_id=document.doc_id,
                    company=document.company,
                    ticker=document.ticker,
                    fiscal_year=document.fiscal_year,
                    section_title=title,
                    text=f"[{document.company} ({document.ticker}) | {title} | Page {page}]\n{chunk_text}",
                    token_count_approx=int(len(chunk_text.split()) * 1.3),
                    page_number=page
                ))
                chunk_idx += 1

        return chunks


if __name__ == "__main__":
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 70)
    print("[STEP 2 DEMO] Section-Aware Chunking")
    print("=" * 70)
    ingestion = DocumentIngestionEngine()
    docs = ingestion.load_all_reports(REPORTS_DIR)
    chunker = SectionAwareFinancialChunker()

    all_chunks = []
    for doc in docs:
        c = chunker.chunk_document(doc)
        all_chunks.extend(c)
        print(f"  * {doc.company}: {len(c)} chunks generated")

    print(f"\nTotal Chunks Across Corpus: {len(all_chunks)}")
    if all_chunks:
        print("\n--- Sample Chunk Preview (First 250 Chars) ---")
        sample = all_chunks[0]
        print(f"Chunk ID: {sample.chunk_id}")
        print(f"Metadata: Company={sample.company}, Section={sample.section_title}, Page={sample.page_number}")
        print(f"Text:\n{sample.text[:250]}...\n")
    print("[OK] Chunking module operational.")
