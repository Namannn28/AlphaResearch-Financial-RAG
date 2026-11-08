"""
==============================================================================
QUERY PREPROCESSOR & SPELLING CORRECTION ENGINE (src/query_preprocessor.py)
==============================================================================
Role in RAG Pipeline:
    Preprocesses incoming user queries before retrieval:
    1. Greeting & Small-Talk Detection: Detects greetings (hi, hello, hey) and
       answers politely without triggering document retrieval refusals.
    2. Typo & Spelling Correction: Automatically corrects misspellings like
       "aple" -> "apple", "tesls" -> "tesla", "microsft" -> "microsoft",
       "amazn" -> "amazon", "gogle" -> "google", "revnue" -> "revenue".
    3. "Did you mean?" suggestions: Formats helpful clarification banners.
    4. Company Entity Resolution: Detects target companies to guide retrieval.

Run directly to test:
    python src/query_preprocessor.py
==============================================================================
"""

import sys
from pathlib import Path

# Add project root to path for direct module execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import re
import difflib


# Target vocabulary for fuzzy matching
KNOWN_COMPANIES = {
    "apple": "Apple Inc.",
    "aapl": "Apple Inc.",
    "iphone": "Apple Inc.",
    "microsoft": "Microsoft Corporation",
    "msft": "Microsoft Corporation",
    "azure": "Microsoft Corporation",
    "tesla": "Tesla, Inc.",
    "tsla": "Tesla, Inc.",
    "amazon": "Amazon.com, Inc.",
    "amzn": "Amazon.com, Inc.",
    "aws": "Amazon.com, Inc.",
    "alphabet": "Alphabet Inc.",
    "googl": "Alphabet Inc.",
    "google": "Alphabet Inc.",
    "goog": "Alphabet Inc."
}

KNOWN_FINANCIAL_TERMS = [
    "apple", "microsoft", "tesla", "amazon", "alphabet", "google",
    "revenue", "financial", "financials", "earnings", "sales", "margin",
    "margins", "profit", "operating", "income", "expenses", "deliveries",
    "capex", "expenditures", "services", "growth", "balance", "guidance"
]

COMMON_MISSPELLINGS = {
    "aple": "apple",
    "appl": "apple",
    "applee": "apple",
    "microsft": "microsoft",
    "micrsoft": "microsoft",
    "micro soft": "microsoft",
    "msft": "msft",
    "tesls": "tesla",
    "teslas": "tesla",
    "tsla": "tsla",
    "amazn": "amazon",
    "amazone": "amazon",
    "amzn": "amzn",
    "alphabt": "alphabet",
    "alphabett": "alphabet",
    "gogle": "google",
    "googl": "googl",
    "revnue": "revenue",
    "revenu": "revenue",
    "revenus": "revenue",
    "finacial": "financial",
    "finacials": "financials",
    "finances": "financials",
    "margn": "margin",
    "proft": "profit",
    "delivries": "deliveries",
    "delvery": "delivery"
}

GREETING_WORDS = {
    "hi", "hello", "hey", "greetings", "good morning", "good afternoon",
    "good evening", "howdy", "hola", "sup", "yo", "namaste", "whatsup",
    "who are you", "what can you do", "help", "how are you"
}


@dataclass
class PreprocessedQuery:
    """Encapsulates the preprocessed query, typo corrections, and intent."""
    raw_query: str
    corrected_query: str
    is_greeting: bool
    greeting_response: Optional[str] = None
    corrections: Dict[str, str] = field(default_factory=dict)
    detected_company: Optional[str] = None
    did_you_mean: Optional[str] = None


class QueryPreprocessor:
    """Preprocesses user queries for typos, greetings, and company entities."""

    def __init__(self):
        self.known_targets = KNOWN_FINANCIAL_TERMS

    def check_greeting(self, text: str) -> Tuple[bool, Optional[str]]:
        """Checks if input is a greeting or introductory inquiry."""
        clean = re.sub(r"[^a-zA-Z\s]", "", text).strip().lower()
        if not clean:
            return False, None

        if clean in GREETING_WORDS or any(clean.startswith(f"{g} ") for g in ["hi", "hello", "hey", "good morning", "good evening"]):
            polite_greeting = (
                "Hello! 👋 I am your AI Financial Analyst.\n\n"
                "I can help you analyze financial disclosures, extract key metrics, evaluate performance, "
                "and review any financial reports or PDFs you provide.\n\n"
                "How can I assist you today?"
            )
            return True, polite_greeting

        return False, None

    def correct_spelling(self, text: str) -> Tuple[str, Dict[str, str]]:
        """Detects and corrects spelling errors using exact dictionary and difflib."""
        tokens = re.findall(r"\b[\$]?[a-zA-Z0-9_\.]+%?\b", text)
        corrected_tokens = []
        corrections = {}

        for tok in tokens:
            tok_clean = tok.lower().strip("$,%.?!")
            # 1. Exact common misspellings
            if tok_clean in COMMON_MISSPELLINGS:
                replacement = COMMON_MISSPELLINGS[tok_clean]
                corrected_tokens.append(replacement)
                if replacement != tok_clean:
                    corrections[tok] = replacement
                continue

            # 2. Fuzzy match against known financial vocabulary (cutoff 0.70)
            if len(tok_clean) >= 4 and tok_clean not in KNOWN_FINANCIAL_TERMS:
                matches = difflib.get_close_matches(tok_clean, self.known_targets, n=1, cutoff=0.72)
                if matches and matches[0] != tok_clean:
                    replacement = matches[0]
                    corrected_tokens.append(replacement)
                    corrections[tok] = replacement
                    continue

            corrected_tokens.append(tok)

        corrected_query = " ".join(corrected_tokens)
        return corrected_query, corrections

    def detect_company(self, query: str) -> Optional[str]:
        """Detects if query targets a specific company."""
        words = re.findall(r"\b[a-zA-Z]+\b", query.lower())
        for w in words:
            if w in KNOWN_COMPANIES:
                return KNOWN_COMPANIES[w]
        return None

    def preprocess(self, raw_query: str) -> PreprocessedQuery:
        """Runs complete query preprocessing flow."""
        # 1. Check for greeting
        is_greet, greeting_msg = self.check_greeting(raw_query)
        if is_greet:
            return PreprocessedQuery(
                raw_query=raw_query,
                corrected_query=raw_query,
                is_greeting=True,
                greeting_response=greeting_msg
            )

        # 2. Correct spelling
        corrected, corrections = self.correct_spelling(raw_query)

        # 3. Detect company
        company = self.detect_company(corrected)

        # 4. Build "Did you mean?" banner
        did_you_mean = None
        if corrections:
            changes = [f"**{target}** (instead of *{orig}*)" for orig, target in corrections.items()]
            did_you_mean = f"Did you mean: {', '.join(changes)}?"

        return PreprocessedQuery(
            raw_query=raw_query,
            corrected_query=corrected,
            is_greeting=False,
            corrections=corrections,
            detected_company=company,
            did_you_mean=did_you_mean
        )


if __name__ == "__main__":
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 70)
    print("[DEMO] Query Preprocessor & Spelling Correction Engine")
    print("=" * 70)
    preprocessor = QueryPreprocessor()

    test_queries = [
        "hi",
        "hello there",
        "tell me about financials about aple",
        "what was revnue of tesls in 2023",
        "microsft intelligent cloud sales",
        "amazn aws operating income",
        "gogle search revenue"
    ]

    for q in test_queries:
        res = preprocessor.preprocess(q)
        print(f"\nRaw Query: \"{q}\"")
        if res.is_greeting:
            print(f"  [Intent: GREETING] Response Excerpt: {res.greeting_response.splitlines()[0]}")
        else:
            print(f"  Corrected Query: \"{res.corrected_query}\"")
            if res.corrections:
                print(f"  Corrections: {res.corrections}")
                print(f"  Banner: {res.did_you_mean}")
            if res.detected_company:
                print(f"  Company Detected: {res.detected_company}")

    print("\n[OK] Query preprocessor module operational.")
