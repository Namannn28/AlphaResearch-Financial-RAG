"""
Document Ingestion Module for Company Annual Reports (10-K Filings).
Loads .md, .txt, and .pdf documents and extracts company metadata and sections.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
import re

@dataclass
class FinancialDocument:
    """Represents a loaded company financial disclosure document."""
    doc_id: str
    company: str
    ticker: str
    fiscal_year: str
    file_path: str
    raw_content: str
    sections: List[Dict[str, str]]

class DocumentIngestionEngine:
    """Loads and sanitizes financial reports into structured memory."""

    @staticmethod
    def extract_metadata_from_header(text: str) -> Dict[str, str]:
        """Extracts company name, ticker, and fiscal year from standard report header."""
        meta = {
            "company": "Unknown",
            "ticker": "N/A",
            "fiscal_year": "2023"
        }
        for line in text.split("\n")[:25]:
            line_clean = line.strip().lower()
            if "company:" in line_clean:
                meta["company"] = line.split(":", 1)[1].strip()
            elif "ticker:" in line_clean:
                meta["ticker"] = line.split(":", 1)[1].strip().upper()
            elif "fiscal year:" in line_clean:
                meta["fiscal_year"] = line.split(":", 1)[1].strip()
        return meta

    @staticmethod
    def parse_markdown_sections(text: str) -> List[Dict[str, str]]:
        """
        Splits financial document along Markdown heading boundaries (## Item X).
        Preserves section titles such as Item 1 Business, Item 7 MD&A, and Item 1A Risk Factors.
        """
        # Match headings like "## Item 1. Business Overview" or "### Segment Breakdown"
        heading_pattern = r"(^#{2,3}\s+[^\n]+)"
        tokens = re.split(heading_pattern, text, flags=re.MULTILINE)

        sections: List[Dict[str, str]] = []
        current_title = "Document Overview"

        for token in tokens:
            token_stripped = token.strip()
            if not token_stripped:
                continue

            if token.startswith("## ") or token.startswith("### "):
                current_title = token.lstrip("#").strip()
            else:
                sections.append({
                    "title": current_title,
                    "content": token_stripped
                })

        return sections

    def load_file(self, file_path: Path) -> FinancialDocument:
        """Loads a single financial report file."""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        meta = self.extract_metadata_from_header(content)
        sections = self.parse_markdown_sections(content)

        return FinancialDocument(
            doc_id=file_path.stem,
            company=meta["company"],
            ticker=meta["ticker"],
            fiscal_year=meta["fiscal_year"],
            file_path=str(file_path),
            raw_content=content,
            sections=sections
        )

    def load_all_reports(self, directory: Path) -> List[FinancialDocument]:
        """Loads all supported reports from a folder."""
        documents: List[FinancialDocument] = []
        if not directory.exists():
            return documents

        for file_path in directory.glob("*.md"):
            documents.append(self.load_file(file_path))

        return documents
