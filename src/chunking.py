"""
Chunking Strategies Module for CandidateFit-RAG.

This module implements and compares two distinct document segmentation strategies:
1. Fixed-Window Sliding Chunker: Breaks text into fixed-size word windows with overlap.
2. Section-Aware Semantic Chunker: Segments documents along natural semantic boundaries
   (e.g., Professional Summary, Technical Skills, Specific Work Experience roles, Projects, Education).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
import re

from src.ingestion import Document
from src.config import chunking_config


@dataclass
class Chunk:
    """Represents an indexed text unit with rich metadata for retrieval attribution."""
    chunk_id: str
    doc_id: str
    candidate_name: str
    section_title: str
    text: str
    chunk_strategy: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def citation_label(self) -> str:
        """Formatted human-readable citation for grounded answers."""
        return f"[{self.candidate_name} | {self.section_title}]"


class FixedWindowChunker:
    """
    Standard sliding-window chunker commonly found in introductory RAG tutorials.
    Splits text purely by word counts with an overlap buffer.
    """
    def __init__(self, window_size_words: int = None, overlap_words: int = None):
        self.window_size = window_size_words or chunking_config.fixed_chunk_size_words
        self.overlap = overlap_words or chunking_config.fixed_chunk_overlap_words

        if self.overlap >= self.window_size:
            raise ValueError("Overlap size must be strictly smaller than window size.")

    def chunk_document(self, document: Document) -> List[Chunk]:
        words = document.content.split()
        if not words:
            return []

        chunks: List[Chunk] = []
        step = self.window_size - self.overlap
        chunk_index = 0

        for start_idx in range(0, len(words), step):
            end_idx = min(start_idx + self.window_size, len(words))
            chunk_words = words[start_idx:end_idx]
            chunk_text = " ".join(chunk_words)

            chunk_id = f"{document.doc_id}_fixed_{chunk_index:03d}"
            candidate_name = document.metadata.get("candidate_name", "Unknown Candidate")

            # Extract any proximate section header within the slice for metadata tracking
            header_match = re.search(r"##\s+([^\n]+)", chunk_text)
            section_title = header_match.group(1).strip() if header_match else f"Offset {start_idx}-{end_idx}"

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=document.doc_id,
                    candidate_name=candidate_name,
                    section_title=section_title,
                    text=chunk_text,
                    chunk_strategy="fixed_window",
                    metadata={
                        "word_count": len(chunk_words),
                        "start_word_index": start_idx,
                        "end_word_index": end_idx,
                    }
                )
            )
            chunk_index += 1

            if end_idx >= len(words):
                break

        return chunks


class SectionAwareSemanticChunker:
    """
    Semantic chunker specifically designed for structured resumes and technical portfolios.
    Identifies logical sections (H2 '##' and H3 '###' headers), keeping cohesive accomplishments
    intact rather than arbitrarily slicing sentences mid-bullet.
    
    Prepends candidate identity and section header to each chunk text to enrich embedding representations.
    """
    def __init__(
        self,
        min_section_length_chars: int = None,
        max_section_length_chars: int = None
    ):
        self.min_length = min_section_length_chars or chunking_config.min_section_length_chars
        self.max_length = max_section_length_chars or chunking_config.max_section_length_chars

    def chunk_document(self, document: Document) -> List[Chunk]:
        content = document.content
        candidate_name = document.metadata.get("candidate_name", "Unknown Candidate")
        
        # Split markdown by H2 headers (## Section)
        # Regex captures delimiter so section headers are preserved
        raw_sections = re.split(r"(^##\s+[^\n]+)", content, flags=re.MULTILINE)
        
        chunks: List[Chunk] = []
        current_header = "Overview"
        chunk_index = 0

        # When re.split with a group is used, items alternate between delimiter and section body
        i = 0
        while i < len(raw_sections):
            part = raw_sections[i].strip()
            if not part:
                i += 1
                continue

            if part.startswith("## "):
                current_header = part.replace("## ", "").strip()
                i += 1
                body = raw_sections[i].strip() if i < len(raw_sections) else ""
            else:
                body = part

            # If body has sub-roles (### Sub-header), split into sub-chunks
            sub_sections = re.split(r"(^###\s+[^\n]+)", body, flags=re.MULTILINE)
            
            if len(sub_sections) > 1:
                sub_header = current_header
                j = 0
                while j < len(sub_sections):
                    sub_part = sub_sections[j].strip()
                    if not sub_part:
                        j += 1
                        continue
                    if sub_part.startswith("### "):
                        sub_header = f"{current_header} - {sub_part.replace('### ', '').strip()}"
                        j += 1
                        sub_body = sub_sections[j].strip() if j < len(sub_sections) else ""
                    else:
                        sub_body = sub_part

                    if len(sub_body) >= self.min_length:
                        chunk_text = f"Candidate: {candidate_name}\nSection: {sub_header}\n\n{sub_body}"
                        chunks.append(
                            Chunk(
                                chunk_id=f"{document.doc_id}_sec_{chunk_index:03d}",
                                doc_id=document.doc_id,
                                candidate_name=candidate_name,
                                section_title=sub_header,
                                text=chunk_text,
                                chunk_strategy="section_aware",
                                metadata={"char_count": len(chunk_text), "header": sub_header}
                            )
                        )
                        chunk_index += 1
                    j += 1
            else:
                if len(body) >= self.min_length:
                    chunk_text = f"Candidate: {candidate_name}\nSection: {current_header}\n\n{body}"
                    chunks.append(
                        Chunk(
                            chunk_id=f"{document.doc_id}_sec_{chunk_index:03d}",
                            doc_id=document.doc_id,
                            candidate_name=candidate_name,
                            section_title=current_header,
                            text=chunk_text,
                            chunk_strategy="section_aware",
                            metadata={"char_count": len(chunk_text), "header": current_header}
                        )
                    )
                    chunk_index += 1

            i += 1

        return chunks


def build_chunks(
    documents: List[Document],
    strategy: str = "section_aware"
) -> List[Chunk]:
    """
    Factory function to segment a list of documents using the requested strategy.
    
    Supported strategies:
    - 'section_aware': Preserves resume domain semantics and heading structures.
    - 'fixed_window': Baseline sliding word-window segmentation.
    """
    if strategy == "section_aware":
        chunker = SectionAwareSemanticChunker()
    elif strategy == "fixed_window":
        chunker = FixedWindowChunker()
    else:
        raise ValueError(f"Unknown chunking strategy '{strategy}'. Choose 'section_aware' or 'fixed_window'.")

    all_chunks: List[Chunk] = []
    for doc in documents:
        doc_chunks = chunker.chunk_document(doc)
        all_chunks.extend(doc_chunks)

    return all_chunks
