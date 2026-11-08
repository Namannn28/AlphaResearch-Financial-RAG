"""
Document Chunking Module for Financial Research Assistant.
Section-aware and Page-aware chunking preserving tables and metrics.
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
        chunks: List[FinancialChunk] = []
        chunk_idx = 0

        for sec in document.sections:
            title = sec["title"]
            content = sec["content"]
            page = sec.get("page", 1)

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
