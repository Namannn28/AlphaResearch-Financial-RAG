"""
LLM Client Integration for Financial Research Assistant using LangChain & Mistral AI.
Supports:
1. LangChain ChatMistralAI (`from langchain_mistralai import ChatMistralAI`)
2. Groq Cloud API (`llama-3.3-70b-versatile` via free Groq API key)
3. Offline Grounded Engine (Runs without an API key for instant offline testing & interviews)
"""

from typing import List, Dict, Any, Optional
import os
import re
import json
import urllib.request
import urllib.error

# LangChain Imports for Mistral AI
try:
    from langchain_mistralai import ChatMistralAI
    from langchain_core.messages import SystemMessage, HumanMessage
    HAS_LANGCHAIN_MISTRAL = True
except ImportError:
    HAS_LANGCHAIN_MISTRAL = False

class FinancialLLMClient:
    """
    Orchestrates LLM generation with strict grounding and source citations.
    Integrates LangChain's ChatMistralAI with seamless offline fallback.
    """

    SYSTEM_PROMPT = """You are an expert Senior Financial Analyst & Research Assistant specializing in SEC 10-K filings.
Your job is to answer investor queries accurately using ONLY the provided verified financial filing contexts.

CRITICAL INSTRUCTIONS:
1. Only answer based on facts directly documented in the context. Never speculate or hallucinate.
2. If the context does not contain sufficient information to answer the question, state:
   "[REFUSAL - INSUFFICIENT_EVIDENCE]: The provided 10-K filings do not contain information regarding this topic."
3. Explicitly cite the company, ticker, and section for each metric mentioned (e.g. [Apple (AAPL) | Item 7 MD&A]).
4. Be precise with financial numbers, percentages, and currencies ($ billions/millions).
5. For multi-company questions, provide a structured comparative analysis."""

    def __init__(
        self,
        mistral_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        provider: str = "mistral"
    ):
        self.mistral_api_key = mistral_api_key or os.getenv("MISTRAL_API_KEY", "")
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
        self.provider = provider.lower()
        self._mistral_chat_model = None

        if self.mistral_api_key and HAS_LANGCHAIN_MISTRAL:
            self._init_langchain_mistral()

    def _init_langchain_mistral(self):
        """Initializes LangChain ChatMistralAI model with user's API key."""
        try:
            self._mistral_chat_model = ChatMistralAI(
                model="mistral-small-latest",
                mistral_api_key=self.mistral_api_key,
                temperature=0.1,
                max_tokens=600
            )
        except Exception:
            self._mistral_chat_model = None

    def set_keys(self, mistral_key: Optional[str] = None, groq_key: Optional[str] = None):
        """Allows updating API keys dynamically from the Streamlit UI."""
        if mistral_key:
            self.mistral_api_key = mistral_key.strip()
            if HAS_LANGCHAIN_MISTRAL:
                self._init_langchain_mistral()
        if groq_key:
            self.groq_api_key = groq_key.strip()

    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        confidence_score: float,
        confidence_threshold: float = 0.28
    ) -> Dict[str, Any]:
        """
        Synthesizes grounded response from retrieved chunks using LangChain ChatMistralAI.
        Triggers explicit safety refusal if confidence is below threshold.
        """
        # Guardrail Refusal Check
        if not retrieved_chunks or confidence_score < confidence_threshold:
            return {
                "answer": (
                    f"[REFUSAL - INSUFFICIENT_EVIDENCE]: The indexed 10-K filings do not contain documented "
                    f"evidence to reliably answer: \"{query}\". "
                    f"(Confidence score {confidence_score:.2f} is below the safety threshold of {confidence_threshold:.2f}). "
                    f"Refusing to generate an ungrounded response to prevent financial hallucination."
                ),
                "is_refusal": True,
                "provider_used": "GuardrailRefusal",
                "citations": []
            }

        # Build Structured Context Prompt
        context_blocks = []
        citations = []
        for i, hit in enumerate(retrieved_chunks, start=1):
            chunk = hit["chunk"]
            citation_str = f"{chunk.company} ({chunk.ticker}) - {chunk.section_title}"
            citations.append(citation_str)
            context_blocks.append(f"--- Context [{i}]: {citation_str} ---\n{chunk.text}")

        combined_context = "\n\n".join(context_blocks)
        user_prompt = f"""Context from verified 10-K filings:
{combined_context}

Question: {query}

Provide a concise, professional financial analysis answering the question, with exact numbers and citations:"""

        # Provider Dispatch: LangChain ChatMistralAI
        if self.provider == "mistral" and self.mistral_api_key:
            try:
                if HAS_LANGCHAIN_MISTRAL:
                    if self._mistral_chat_model is None:
                        self._init_langchain_mistral()

                    messages = [
                        SystemMessage(content=self.SYSTEM_PROMPT),
                        HumanMessage(content=user_prompt)
                    ]
                    response = self._mistral_chat_model.invoke(messages)
                    answer = response.content.strip()
                    return {
                        "answer": answer,
                        "is_refusal": False,
                        "provider_used": "LangChain (ChatMistralAI)",
                        "citations": citations
                    }
                else:
                    # Direct HTTP fallback to Mistral API
                    answer = self._call_mistral_direct_api(user_prompt)
                    return {
                        "answer": answer,
                        "is_refusal": False,
                        "provider_used": "Mistral AI (Direct API)",
                        "citations": citations
                    }
            except Exception as e:
                # If Mistral API call fails or key invalid, cleanly fallback to offline synthesis
                offline_answer = self._offline_extractive_synthesis(query, retrieved_chunks)
                return {
                    "answer": f"{offline_answer}\n\n*(Note: LangChain ChatMistralAI encountered error: {e}. Fallback synthesized from verified chunks.)*",
                    "is_refusal": False,
                    "provider_used": "Offline Fallback",
                    "citations": citations
                }

        # Provider Dispatch: Groq API
        elif self.provider == "groq" and self.groq_api_key:
            try:
                answer = self._call_groq_api(user_prompt)
                return {
                    "answer": answer,
                    "is_refusal": False,
                    "provider_used": "Groq Cloud (API)",
                    "citations": citations
                }
            except Exception as e:
                offline_answer = self._offline_extractive_synthesis(query, retrieved_chunks)
                return {
                    "answer": f"{offline_answer}\n\n*(Note: Groq API encountered: {e}. Fallback synthesized from verified chunks.)*",
                    "is_refusal": False,
                    "provider_used": "Offline Fallback",
                    "citations": citations
                }

        # Default: Deterministic Offline Grounded Extractive Synthesis
        answer = self._offline_extractive_synthesis(query, retrieved_chunks)
        return {
            "answer": answer,
            "is_refusal": False,
            "provider_used": "Offline Grounded Engine",
            "citations": citations
        }

    def _call_mistral_direct_api(self, user_prompt: str) -> str:
        """Invokes Mistral AI API directly via standard Python urllib."""
        url = "https://api.mistral.ai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.mistral_api_key}"
        }
        payload = {
            "model": "mistral-small-latest",
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

    def _call_groq_api(self, user_prompt: str) -> str:
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
            f"Based on verified disclosures in the **{chunk.company} ({chunk.ticker})** 10-K filing "
            f"([*{chunk.section_title}*]):\n\n"
            f"{formatted_quotes}\n\n"
            f"**Source Document**: `{chunk.doc_id}.md` (Section: *{chunk.section_title}*)"
        )
