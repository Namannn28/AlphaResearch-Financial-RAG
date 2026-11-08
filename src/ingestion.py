"""
Document Ingestion & Text Preprocessing Module.

This module handles reading candidate resumes and job specifications from disk,
normalizing text formatting, and attaching structural metadata.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any
import re

from src.config import ingestion_config


@dataclass
class Document:
    """Represents an ingested raw document prior to segmentation/chunking."""
    doc_id: str
    filename: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def word_count(self) -> int:
        """Returns the total word count of the document."""
        return len(self.content.split())


def clean_text(text: str) -> str:
    """
    Normalizes whitespace and strips unnecessary control artifacts
    while preserving meaningful markdown headings (#, ##).
    """
    # Replace non-breaking spaces and Windows CR characters
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = cleaned.replace("\u00a0", " ")

    # Collapse excessive blank lines into double newlines for paragraph preservation
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    # Normalize trailing and leading spaces per line
    lines = [line.strip() for line in cleaned.split("\n")]
    cleaned = "\n".join(lines).strip()

    return cleaned


def extract_candidate_name_from_markdown(content: str, filename: str) -> str:
    """
    Attempts to extract candidate name from the first H1 header in markdown (# Name).
    Falls back to parsing the file name if not explicitly found.
    """
    match = re.search(r"^#\s+([A-Za-z\s\.\-]+)", content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    
    # Fallback to file name extraction (e.g. candidate_1_alex_chen -> Alex Chen)
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) >= 3:
        return " ".join(parts[2:]).title()
    return stem.replace("_", " ").title()


def load_single_document(file_path: Path) -> Document:
    """
    Reads a single markdown or text file and creates a validated Document object.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Document file does not exist at {file_path}")

    with open(file_path, "r", encoding="utf-8", errors="ignore") as file_handle:
        raw_content = file_handle.read()

    cleaned_content = clean_text(raw_content)
    candidate_name = extract_candidate_name_from_markdown(cleaned_content, file_path.name)

    metadata = {
        "candidate_name": candidate_name,
        "source_path": str(file_path),
        "extension": file_path.suffix.lower(),
        "char_count": len(cleaned_content)
    }

    return Document(
        doc_id=file_path.stem,
        filename=file_path.name,
        content=cleaned_content,
        metadata=metadata
    )


def load_corpus_from_directory(directory_path: Path) -> List[Document]:
    """
    Iterates over a directory and returns a sorted list of validated Document objects.
    """
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    loaded_documents: List[Document] = []
    
    # Sort files to guarantee deterministic indexing order across platforms
    file_paths = sorted(directory_path.glob("*.*"))

    for file_path in file_paths:
        if file_path.suffix.lower() in ingestion_config.allowed_extensions:
            document = load_single_document(file_path)
            loaded_documents.append(document)

    return loaded_documents
