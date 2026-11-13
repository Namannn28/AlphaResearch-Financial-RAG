"""
==============================================================================
STEP 6: LLM GENERATION & CITATION CLIENT (src/llm_client.py)
==============================================================================
Role in RAG Pipeline:
    Synthesizes concise, grounded financial answers with exact page/section
    citations using LangChain's ChatGroq client. Enforces explicit refusal
    when retrieval confidence falls below the safety threshold.

Key Architectural & Design Principles:
    - LangChain ChatGroq: Fast hosted inference via Groq's LPU architecture,
      running active models (e.g. llama-3.3-70b-versatile, llama-3.1-8b-instant).
    - Grounding & Guardrails: System prompt forbids financial speculation. If
      confidence < 0.28, returns structured refusal instead of hallucinating.
    - Zero-Crash Offline Fallback: If no API key is supplied or rate limits
      hit, deterministically synthesizes the answer from retrieved chunks.

Run directly to test:
    python src/llm_client.py
==============================================================================
"""

import sys
from pathlib import Path

# Add project root to path for direct module execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import List, Dict, Any, Optional
import os
import re
import json
import urllib.request
import urllib.error
from dotenv import load_dotenv

# Load .env
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# LangChain Imports for Groq Cloud
try:
    from langchain_groq import ChatGroq
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    HAS_LANGCHAIN_GROQ = True
except ImportError:
    HAS_LANGCHAIN_GROQ = False


class FinancialLLMClient:
    """
    Orchestrates LLM generation with strict grounding and source citations.
    Integrates LangChain's ChatGroq with seamless offline fallback.
    """

    SYSTEM_PROMPT = """You are an expert Senior Financial Analyst & Research Assistant specializing in SEC 10-K filings.
Your job is to answer investor queries accurately using ONLY the provided verified financial filing contexts.

CRITICAL INSTRUCTIONS:
1. Only answer based on facts directly documented in the context. Never speculate or hallucinate.
2. If the context does not contain sufficient information to answer the question, state:
   "[REFUSAL - INSUFFICIENT_EVIDENCE]: The provided document excerpts do not contain documented evidence regarding this topic."
3. Explicitly cite the company, ticker, and section for each metric mentioned (e.g. [Apple (AAPL) | Item 7 MD&A]).
4. Be precise with financial numbers, percentages, and currencies ($ billions/millions).
5. For multi-company questions, provide a structured comparative analysis."""

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        provider: str = "groq"
    ):
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
        self.provider = provider.lower()
        self._groq_chat_model = None

        if self.groq_api_key and HAS_LANGCHAIN_GROQ:
            self._init_langchain_groq()

    def _init_langchain_groq(self):
        """Initializes LangChain ChatGroq model with user's free API key."""
        models_to_try = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it"
        ]
        for m in models_to_try:
            try:
                self._groq_chat_model = ChatGroq(
                    model=m,
                    groq_api_key=self.groq_api_key,
                    temperature=0.1,
                    max_tokens=600
                )
                self.active_model_name = m
                break
            except Exception:
                continue

    def set_keys(self, groq_key: Optional[str] = None):
        """Allows updating API keys dynamically from the Streamlit UI."""
        if groq_key:
            self.groq_api_key = groq_key.strip()
            if HAS_LANGCHAIN_GROQ:
                self._init_langchain_groq()

    @staticmethod
    def _sanitize_output_text(text: str) -> str:
        """Normalizes unicode characters to safe characters to prevent Windows console encoding errors."""
        replacements = {
            "\u202f": " ",
            "\xa0": " ",
            "\u200b": "",
            "’": "'",
            "‘": "'",
            "“": '"',
            "”": '"',
            "≈": "~",
            "–": "-",
            "—": "-",
            "•": "*",
            "…": "...",
            "【": "[",
            "】": "]"
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        return text

    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        confidence_score: float,
        confidence_threshold: float = 0.28,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Synthesizes grounded response from retrieved chunks using LangChain ChatGroq.
        Triggers explicit safety refusal if confidence is below threshold.
        Maintains conversational history for the active session.
        """
        # Guardrail Refusal Check
        if not retrieved_chunks or confidence_score < confidence_threshold:
            refusal_msg = (
                f"[REFUSAL - INSUFFICIENT_EVIDENCE]: The indexed documents do not contain documented "
                f"evidence to reliably answer: \"{query}\". "
                f"(Confidence score {confidence_score:.2f} is below the safety threshold of {confidence_threshold:.2f}). "
                f"Refusing to generate an ungrounded response to prevent financial hallucination."
            )
            return {
                "answer": self._sanitize_output_text(refusal_msg),
                "is_refusal": True,
                "provider_used": "GuardrailRefusal",
                "citations": []
            }

        # Build Structured Context Prompt
        context_blocks = []
        citations = []
        for i, hit in enumerate(retrieved_chunks, start=1):
            chunk = hit["chunk"]
            citation_str = f"{chunk.company} ({chunk.ticker}) - {chunk.section_title} | Page {chunk.page_number}"
            citations.append(citation_str)
            context_blocks.append(f"--- Context [{i}]: {citation_str} ---\n{chunk.text}")

        combined_context = "\n\n".join(context_blocks)
        user_prompt = f"""Context from verified documents:
{combined_context}

Question: {query}

Provide a concise, professional financial analysis answering the question, with exact numbers and citations:"""

        # Provider Dispatch: LangChain ChatGroq
        if (self.provider == "groq" or not self.provider) and self.groq_api_key:
            try:
                if HAS_LANGCHAIN_GROQ:
                    if self._groq_chat_model is None:
                        self._init_langchain_groq()

                    if self._groq_chat_model is not None:
                        messages = [SystemMessage(content=self.SYSTEM_PROMPT)]
                        if conversation_history:
                            # Append recent turns from this active session only
                            for turn in conversation_history[-6:]:
                                role = turn.get("role")
                                text = turn.get("content", "")
                                if role == "user":
                                    messages.append(HumanMessage(content=text))
                                elif role == "assistant" and not turn.get("is_refusal", False):
                                    messages.append(AIMessage(content=text))

                        messages.append(HumanMessage(content=user_prompt))
                        response = self._groq_chat_model.invoke(messages)
                        answer = self._sanitize_output_text(response.content.strip())
                        model_name = getattr(self, "active_model_name", "ChatGroq")
                        return {
                            "answer": answer,
                            "is_refusal": False,
                            "provider_used": f"LangChain ChatGroq ({model_name})",
                            "citations": citations
                        }

                # Direct API fallback
                answer = self._sanitize_output_text(self._call_groq_direct_api(user_prompt))
                return {
                    "answer": answer,
                    "is_refusal": False,
                    "provider_used": "Groq Cloud (Direct API)",
                    "citations": citations
                }
            except Exception as e:
                offline_answer = self._sanitize_output_text(self._offline_extractive_synthesis(query, retrieved_chunks))
                return {
                    "answer": f"{offline_answer}\n\n*(Note: Cloud call encountered '{e}'. Fallback synthesized directly from verified chunks.)*",
                    "is_refusal": False,
                    "provider_used": "Offline Fallback",
                    "citations": citations
                }

        # Default: Deterministic Offline Grounded Extractive Synthesis
        answer = self._sanitize_output_text(self._offline_extractive_synthesis(query, retrieved_chunks))
        return {
            "answer": answer,
            "is_refusal": False,
            "provider_used": "Offline Grounded Engine",
            "citations": citations
        }

    def _call_groq_direct_api(self, user_prompt: str) -> str:
        """Invokes Groq API directly via standard Python urllib."""
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.groq_api_key}"
        }
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 500
        }

        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
        with urllib.request.urlopen(req, timeout=15) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            return res_data["choices"][0]["message"]["content"].strip()

    def _offline_extractive_synthesis(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Deterministically extracts verified financial lines matching query terms."""
        top_hit = retrieved_chunks[0]
        chunk = top_hit["chunk"]
        lines = [line.strip() for line in chunk.text.split("\n") if line.strip() and not line.startswith("[")]

        query_words = set(re.findall(r"\b[a-zA-Z0-9_\.]+\b", query.lower()))
        query_words -= {"what", "was", "how", "much", "did", "the", "in", "and", "for", "with", "from", "total"}

        scored_lines = []
        for line in lines:
            line_words = set(re.findall(r"\b[a-zA-Z0-9_\.]+\b", line.lower()))
            overlap = len(query_words.intersection(line_words))
            has_metric = bool(re.search(r"(\$[\d,\.]+|\b\d+\.?\d*%)", line))
            score = overlap + (2 if has_metric else 0)
            scored_lines.append((score, line))

        scored_lines.sort(key=lambda x: x[0], reverse=True)
        best_lines = [line for score, line in scored_lines[:3] if score > 0]

        if not best_lines:
            best_lines = lines[:2]

        formatted_quotes = "\n".join([f"> {line}" for line in best_lines])

        return (
            f"Based on verified disclosures in the **{chunk.company} ({chunk.ticker})** filing "
            f"([*{chunk.section_title}* | Page {chunk.page_number}]):\n\n"
            f"{formatted_quotes}\n\n"
            f"**Source Document**: `{chunk.doc_id}` (Section: *{chunk.section_title}*)"
        )


if __name__ == "__main__":
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 70)
    print("[STEP 6 DEMO] LangChain ChatGroq Generation Client")
    print("=" * 70)
    client = FinancialLLMClient()
    print(f"Has LangChain Groq: {HAS_LANGCHAIN_GROQ}")
    print(f"Has API Key: {bool(client.groq_api_key)}")

    # Mock chunk for demo
    from src.chunking import FinancialChunk
    mock_chunk = FinancialChunk(
        chunk_id="aapl_demo",
        doc_id="apple_2023_10k",
        company="Apple Inc.",
        ticker="AAPL",
        fiscal_year="2023",
        section_title="Consolidated Financial Performance",
        text="[Apple Inc. (AAPL) | Consolidated Financial Performance | Page 1]\nTotal net sales for fiscal 2023 were $383,285 million compared to $394,328 million in 2022.",
        token_count_approx=40,
        page_number=1
    )
    mock_hits = [{"chunk": mock_chunk, "score": 0.92}]

    print("\nGenerating answer with high confidence (0.92):")
    res = client.generate_answer("What was Apple's total net sales in fiscal 2023?", mock_hits, 0.92)
    print(f"Provider: {res['provider_used']}")
    print(f"Answer:\n{res['answer']}\n")

    print("-" * 50)
    print("Testing safety refusal with low confidence (0.12 < threshold 0.28):")
    refusal_res = client.generate_answer("What was Apple's space tourism revenue?", mock_hits, 0.12)
    print(f"Is Refusal: {refusal_res['is_refusal']}")
    print(f"Refusal Message:\n{refusal_res['answer']}\n")

    print("[OK] LLM generation module operational.")
