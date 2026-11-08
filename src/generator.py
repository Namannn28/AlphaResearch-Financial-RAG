"""
Grounded Answer Generation & Hallucination Guardrail Module.

This module enforces strict factual grounding, explicit out-of-corpus refusals,
citation generation, and token cost/latency profiling.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import time
import os
import re

from src.chunking import Chunk
from src.reranker import RerankResult
from src.config import retrieval_config, generation_config


@dataclass
class GenerationResponse:
    """Production response payload containing the answer, attribution, and operational telemetry."""
    answer: str
    is_refusal: bool
    confidence_score: float
    citations: List[Dict[str, str]] = field(default_factory=list)
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    input_tokens_estimated: int = 0
    output_tokens_estimated: int = 0
    estimated_cost_usd: float = 0.0
    model_name: str = "Deterministic-Grounded-Extractor"


SYSTEM_PROMPT = """You are an objective Technical Recruiting Auditor and RAG Intelligence Engine.
Your duty is to answer questions about candidate resumes strictly using the provided verified evidence chunks.

RULES:
1. Ground every claim directly in the provided context. Include candidate name and section citations.
2. If the context does not explicitly document the answer, you MUST state:
   "INSUFFICIENT_EVIDENCE: The candidate corpus does not contain documented evidence regarding [topic]."
3. Do NOT extrapolate, speculate, assume unlisted skills, or hallucinate credentials.
4. Keep the response factual, concise, and professional.
"""


class GroundedGenerator:
    """
    Synthesizes answers with strict boundary checks and cost instrumentation.
    """
    def __init__(
        self,
        refusal_threshold: float = None,
        api_key: Optional[str] = None,
        provider: str = "auto"
    ):
        self.refusal_threshold = refusal_threshold or retrieval_config.confidence_refusal_threshold
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.provider = provider

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (approx 1 token per 4 characters / 0.75 words)."""
        return max(1, int(len(text) / 4))

    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculates estimated cost in USD based on standard lightweight model pricing."""
        input_cost = (prompt_tokens / 1000.0) * generation_config.estimated_input_cost_per_1k_tokens
        output_cost = (completion_tokens / 1000.0) * generation_config.estimated_output_cost_per_1k_tokens
        return round(input_cost + output_cost, 6)

    def generate(
        self,
        query: str,
        reranked_results: List[RerankResult],
        retrieval_latency_ms: float = 0.0
    ) -> GenerationResponse:
        """
        Synthesizes an attributed response or executes an explicit refusal if confidence is below threshold.
        """
        start_time = time.perf_counter()

        # Handle empty retrieval
        if not reranked_results:
            gen_latency = (time.perf_counter() - start_time) * 1000.0
            return GenerationResponse(
                answer="INSUFFICIENT_EVIDENCE: No matching records were retrieved from the candidate corpus.",
                is_refusal=True,
                confidence_score=0.0,
                citations=[],
                retrieval_latency_ms=retrieval_latency_ms,
                generation_latency_ms=gen_latency,
                total_latency_ms=retrieval_latency_ms + gen_latency,
                input_tokens_estimated=0,
                output_tokens_estimated=18,
                estimated_cost_usd=0.000002,
                model_name="Safety-Guardrail"
            )

        top_result = reranked_results[0]
        top_score = top_result.rerank_score

        # Check refusal guardrail threshold
        # If top similarity or contextual overlap is weak, refuse to hallucinate
        if top_score < self.refusal_threshold:
            gen_latency = (time.perf_counter() - start_time) * 1000.0
            refusal_message = (
                f"INSUFFICIENT_EVIDENCE: The candidate corpus does not contain sufficient documented "
                f"evidence to answer '{query}'. (Confidence score {top_score:.2f} is below the required "
                f"threshold of {self.refusal_threshold:.2f}). Refusing to speculate to avoid hallucination."
            )
            return GenerationResponse(
                answer=refusal_message,
                is_refusal=True,
                confidence_score=float(top_score),
                citations=[],
                retrieval_latency_ms=retrieval_latency_ms,
                generation_latency_ms=gen_latency,
                total_latency_ms=retrieval_latency_ms + gen_latency,
                input_tokens_estimated=self._estimate_tokens(query),
                output_tokens_estimated=self._estimate_tokens(refusal_message),
                estimated_cost_usd=0.000005,
                model_name="Refusal-Guardrail"
            )

        # Build citations
        citations = [
            {
                "citation_id": f"[{i+1}]",
                "candidate_name": res.chunk.candidate_name,
                "section": res.chunk.section_title,
                "chunk_id": res.chunk.chunk_id,
                "score": f"{res.rerank_score:.3f}",
                "snippet": res.chunk.text[:220] + "..." if len(res.chunk.text) > 220 else res.chunk.text
            }
            for i, res in enumerate(reranked_results)
        ]

        # Assemble grounded prompt
        context_blocks = []
        for i, res in enumerate(reranked_results):
            context_blocks.append(
                f"--- EVIDENCE CHUNK [{i+1}] ---\n"
                f"Candidate: {res.chunk.candidate_name}\n"
                f"Section: {res.chunk.section_title}\n"
                f"{res.chunk.text}\n"
            )
        full_context = "\n".join(context_blocks)
        user_prompt = f"Context:\n{full_context}\n\nQuestion: {query}\n\nAttributed Answer:"

        # Try API generator if credentials exist, otherwise utilize grounded extractor
        answer_text, model_name = self._synthesize(user_prompt, query, reranked_results)

        gen_latency = (time.perf_counter() - start_time) * 1000.0
        total_latency = retrieval_latency_ms + gen_latency

        prompt_tokens = self._estimate_tokens(SYSTEM_PROMPT + user_prompt)
        completion_tokens = self._estimate_tokens(answer_text)
        cost = self._calculate_cost(prompt_tokens, completion_tokens)

        # Check if the generated text is itself a refusal
        is_refusal = "INSUFFICIENT_EVIDENCE" in answer_text

        return GenerationResponse(
            answer=answer_text,
            is_refusal=is_refusal,
            confidence_score=float(top_score),
            citations=citations,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=gen_latency,
            total_latency_ms=total_latency,
            input_tokens_estimated=prompt_tokens,
            output_tokens_estimated=completion_tokens,
            estimated_cost_usd=cost,
            model_name=model_name
        )

    def _synthesize(
        self,
        full_prompt: str,
        query: str,
        results: List[RerankResult]
    ) -> Tuple[str, str]:
        """Synthesizes text using external LLM or local grounded extractor."""
        # Check Gemini API
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_key)
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(
                    f"{SYSTEM_PROMPT}\n\n{full_prompt}",
                    generation_config={"temperature": generation_config.temperature}
                )
                if response.text:
                    return response.text.strip(), "Gemini-1.5-Flash"
            except Exception as e:
                pass

        # Check OpenAI API
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            try:
                import urllib.request
                import json
                req_data = json.dumps({
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": full_prompt}
                    ],
                    "temperature": generation_config.temperature
                }).encode("utf-8")
                req = urllib.request.Request(
                    "https://api.openai.com/v1/chat/completions",
                    data=req_data,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {openai_key}"
                    }
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    resp_json = json.loads(resp.read().decode("utf-8"))
                    text = resp_json["choices"][0]["message"]["content"]
                    return text.strip(), "GPT-4o-mini"
            except Exception as e:
                pass

        # High-Fidelity Local Grounded Extractor (Deterministic, Zero-API-Cost)
        return self._local_grounded_extract(query, results), "Local-Grounded-Extractor"

    def _local_grounded_extract(self, query: str, results: List[RerankResult]) -> str:
        """
        Extracts and synthesizes the exact evidentiary sentences from the top retrieved chunks,
        attributing each statement to the candidate and section.
        """
        top_chunk = results[0].chunk
        query_words = set(re.findall(r"\b[A-Za-z0-9\.\-\+]+\b", query.lower())) - {
            "what", "which", "who", "where", "how", "does", "did", "is", "was", "the", "a", "an", "candidate", "experience"
        }

        # Find best matching sentences in top chunk
        sentences = re.split(r"(?<=[.!?])\s+", top_chunk.text)
        best_sentences = []
        for s in sentences:
            s_clean = s.strip()
            if not s_clean or s_clean.startswith("Candidate:") or s_clean.startswith("Section:"):
                continue
            words = set(re.findall(r"\b[A-Za-z0-9\.\-\+]+\b", s_clean.lower()))
            overlap = len(words & query_words)
            if overlap > 0:
                best_sentences.append((s_clean, overlap))

        best_sentences.sort(key=lambda x: x[1], reverse=True)
        evidence_text = " ".join([item[0] for item in best_sentences[:2]])

        if not evidence_text:
            # Fallback to key snippet
            clean_lines = [l for l in top_chunk.text.split("\n") if l.strip() and not l.startswith("#")]
            evidence_text = clean_lines[0] if clean_lines else top_chunk.text[:200]

        citation = f"[{top_chunk.candidate_name} | {top_chunk.section_title}]"
        
        return (
            f"Based on verified records for **{top_chunk.candidate_name}** ({citation}):\n\n"
            f"{evidence_text}\n\n"
            f"*(Confidence: {results[0].rerank_score:.2f} | Evidence Chunk 1)*"
        )
