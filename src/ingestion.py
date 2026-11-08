"""
==============================================================================
STEP 1: DOCUMENT INGESTION ENGINE (src/ingestion.py)
==============================================================================
Role in RAG Pipeline:
    Extracts raw text and metadata from SEC 10-K filings and custom documents
    (.pdf, .md, .txt). Normalizes headings, page numbers, and company metadata
    into standardized FinancialDocument objects.

Key Concepts for Interviews:
    - Multi-format ingestion: Extracts text, tables, and page markers from PDFs
      using pypdf, and splits Markdown files along heading hierarchies (Item 1,
      Item 7 MD&A, Item 1A Risk Factors).
    - Metadata extraction: Automatically parses company names, ticker symbols,
      and fiscal years to enable downstream metadata-filtered search.

Run directly to test:
    python src/ingestion.py
==============================================================================
"""

import sys
from pathlib import Path

# Add project root to path for direct module execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import re
import pypdf

from src.config import REPORTS_DIR


@dataclass
class FinancialDocument:
    """Represents a loaded company financial disclosure or uploaded PDF."""
    doc_id: str
    company: str
    ticker: str
    fiscal_year: str
    file_path: str
    raw_content: str
    sections: List[Dict[str, Any]]


class DocumentIngestionEngine:
    """Loads and sanitizes financial reports and uploaded PDFs into structured memory."""

    @staticmethod
    def extract_metadata_from_header(text: str, filename: str) -> Dict[str, str]:
        """Extracts company name, ticker, and fiscal year from header or filename."""
        clean_name = Path(filename).stem.replace("_", " ").title()
        meta = {
            "company": clean_name.split()[0],
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
    def parse_markdown_sections(text: str) -> List[Dict[str, Any]]:
        """Splits document along Markdown heading boundaries (## Item X)."""
        heading_pattern = r"(^#{2,3}\s+[^\n]+)"
        tokens = re.split(heading_pattern, text, flags=re.MULTILINE)

        sections: List[Dict[str, Any]] = []
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
                    "content": token_stripped,
                    "page": 1
                })

        return sections

    def load_pdf(self, file_path: Path) -> FinancialDocument:
        """Extracts text, page numbers, and tables from a PDF document."""
        reader = pypdf.PdfReader(str(file_path))
        full_text_pages = []
        sections: List[Dict[str, Any]] = []

        doc_meta = {
            "company": file_path.stem.replace("_", " ").title().split()[0],
            "ticker": "PDF",
            "fiscal_year": "2023"
        }

        for page_idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if not text.strip():
                continue
            full_text_pages.append(text)

            # Detect headings or treat each page / major paragraph as a section
            lines = text.split("\n")
            first_line = lines[0].strip() if lines else f"Page {page_idx}"
            title = first_line[:60] if len(first_line) > 5 else f"Page {page_idx} Excerpt"

            sections.append({
                "title": f"Page {page_idx} ({title})",
                "content": text.strip(),
                "page": page_idx
            })

        combined_raw = "\n\n--- Page Break ---\n\n".join(full_text_pages)

        if full_text_pages:
            detected = self.extract_metadata_from_header(full_text_pages[0], file_path.name)
            if detected["company"] != "Unknown":
                doc_meta = detected

        return FinancialDocument(
            doc_id=file_path.stem,
            company=doc_meta["company"],
            ticker=doc_meta["ticker"],
            fiscal_year=doc_meta["fiscal_year"],
            file_path=str(file_path),
            raw_content=combined_raw,
            sections=sections
        )

    def load_file(self, file_path: Path) -> FinancialDocument:
        """Loads a document based on its extension (.pdf, .md, or .txt)."""
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return self.load_pdf(file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        meta = self.extract_metadata_from_header(content, file_path.name)
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
        """Loads all supported documents (.md, .pdf, .txt) from a folder."""
        documents: List[FinancialDocument] = []
        if not directory.exists():
            return documents

        for file_path in sorted(directory.iterdir()):
            if file_path.suffix.lower() in [".md", ".pdf", ".txt"]:
                try:
                    documents.append(self.load_file(file_path))
                except Exception as e:
                    print(f"Warning: Failed to load {file_path.name}: {e}")

        return documents


if __name__ == "__main__":
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 70)
    print("[STEP 1 DEMO] Ingesting Financial Reports")
    print("=" * 70)
    engine = DocumentIngestionEngine()
    docs = engine.load_all_reports(REPORTS_DIR)
    print(f"Successfully loaded {len(docs)} documents from {REPORTS_DIR}:\n")
    for doc in docs:
        print(f"  * {doc.company} ({doc.ticker}) | FY {doc.fiscal_year} | {len(doc.sections)} sections parsed")
        for s in doc.sections[:2]:
            print(f"      - {s['title']} ({len(s['content'])} chars)")
    print("\n[OK] Ingestion module operational.")
