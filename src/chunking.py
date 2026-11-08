"""
Document Chunking Module for Financial Research Assistant.
Compares standard Fixed-Window chunking vs. Section-Aware Financial chunking.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
from src.ingestion import FinancialDocument

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

class SectionAwareFinancialChunker:
    """
    Chunks financial documents respecting section boundaries (e.g. Item 7 MD&A,
    Revenue Breakdown, Risk Factors). Ensures financial numbers remain tied
    to their company context and section title.
    """

    def __init__(self, target_chunk_words: int = 250, min_chars: int = 30):
        self.target_chunk_words = target_chunk_words
        self.min_chars = min_chars

    def chunk_document(self, document: FinancialDocument) -> List[FinancialChunk]:
        chunks: List[FinancialChunk] = []
        chunk_idx = 0

        for sec in document.sections:
            title = sec["title"]
            content = sec["content"]

            if len(content.strip()) < self.min_chars:
                continue

            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
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
                        text=f"[{document.company} ({document.ticker}) | {title}]\n{chunk_text}",
                        token_count_approx=int(len(chunk_text.split()) * 1.3)
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
                    text=f"[{document.company} ({document.ticker}) | {title}]\n{chunk_text}",
                    token_count_approx=int(len(chunk_text.split()) * 1.3)
                ))
                chunk_idx += 1

        return chunks


class FixedWindowChunker:
    """
    Standard naive sliding window chunker (used as baseline in evaluation).
    Splits text blindly without awareness of tables, headers, or company context.
    """

    def __init__(self, window_size_words: int = 200, overlap_words: int = 40):
        self.window_size_words = window_size_words
        self.overlap_words = overlap_words

    def chunk_document(self, document: FinancialDocument) -> List[FinancialChunk]:
        words = document.raw_content.split()
        chunks: List[FinancialChunk] = []
        step = max(1, self.window_size_words - self.overlap_words)

        for i in range(0, len(words), step):
            window = words[i:i + self.window_size_words]
            if len(window) < 15:
                continue

            chunk_text = " ".join(window)
            chunks.append(FinancialChunk(
                chunk_id=f"{document.doc_id}_fixed_{len(chunks)}",
                doc_id=document.doc_id,
                company=document.company,
                ticker=document.ticker,
                fiscal_year=document.fiscal_year,
                section_title="Fixed Window",
                text=chunk_text,
                token_count_approx=int(len(window) * 1.3)
            ))

        return chunks
