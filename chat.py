"""
==============================================================================
chat.py - Interactive Financial & PDF RAG Terminal Assistant
==============================================================================
Run in your terminal:
    python chat.py
    python chat.py --pdf path/to/your_file.pdf

Features:
- Ask questions directly through your terminal CLI
- Drag & drop local PDFs, Markdown, or text files directly into the prompt
- Multi-turn conversation history stored in a messages dictionary
- Customizable system prompt
- Hybrid RAG retrieval (Dense vector + BM25) with refusal guardrail
- High-speed LangChain ChatGroq generation with source citations
- Commands: 'history', 'clear', 'prompt', 'load <file>', 'exit'
==============================================================================
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# 1. UTF-8 console output for Windows terminals
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 2. Load .env environment variables
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# 3. Import RAG components
from src.config import config, REPORTS_DIR
from src.pipeline import FinancialRAGPipeline

# 4. LangChain Groq imports
try:
    from langchain_groq import ChatGroq
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False

DEFAULT_SYSTEM_PROMPT = """You are an expert Senior Financial Analyst & Research Assistant specializing in SEC 10-K filings and corporate documents.
Your role is to assist analysts, investors, and recruiters by providing accurate, evidence-backed answers using verified document excerpt contexts.

OPERATIONAL RULES:
1. Base your answers strictly on the provided document excerpts and conversation history. Never speculate or hallucinate.
2. If the context does not contain enough information to answer, state:
   "[REFUSAL - INSUFFICIENT_EVIDENCE]: The provided document excerpts do not contain documented evidence regarding this inquiry."
3. Always include citations: mention the company, ticker, and filing section or page (e.g., [Apple (AAPL) | Item 7 MD&A]).
4. Maintain a professional, quantitative, analytical tone with exact figures."""

class InteractiveFinancialChatPipeline:
    """Terminal-based interactive RAG pipeline with message history and PDF drag-and-drop."""

    def __init__(self, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        self.system_prompt = system_prompt
        self.groq_api_key = os.getenv("GROQ_API_KEY", "").strip()

        # Message History Dictionary
        self.conversation_history: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]

        print("\n" + "=" * 65)
        print("📈 AlphaResearch Financial & PDF RAG Engine")
        print("=" * 65)

        self.rag_pipeline = FinancialRAGPipeline(
            use_hybrid=True,
            use_reranker=True,
            prefer_neural=True,
            llm_provider="groq"
        )
        self.rag_pipeline.load_and_index(REPORTS_DIR)
        print(f"Indexed {len(self.rag_pipeline.chunks)} initial chunks across Apple, Microsoft, Tesla, Alphabet, Amazon.")

        # Initialize LangChain ChatGroq
        self.chat_model = None
        if self.groq_api_key and HAS_LANGCHAIN:
            try:
                self.chat_model = ChatGroq(
                    model="openai/gpt-oss-120b",
                    groq_api_key=self.groq_api_key,
                    temperature=0.1,
                    max_tokens=700
                )
                print("LangChain ChatGroq connected successfully (Model: openai/gpt-oss-120b).")
            except Exception:
                try:
                    self.chat_model = ChatGroq(
                        model="openai/gpt-oss-20b",
                        groq_api_key=self.groq_api_key,
                        temperature=0.1,
                        max_tokens=700
                    )
                    print("LangChain ChatGroq connected successfully (Model: openai/gpt-oss-20b).")
                except Exception as e:
                    print(f"Notice: Groq fallback ({e}). Running in offline extractive mode.")
        else:
            print("Notice: No GROQ_API_KEY found in .env. Running in deterministic Offline Fallback mode.")

    def load_custom_pdf(self, file_path_str: str) -> bool:
        """Loads and indexes a user-provided PDF, Markdown, or text file."""
        # Clean quotes if user dragged and dropped path into terminal
        cleaned_path = file_path_str.strip().strip("'\"")
        p = Path(cleaned_path)
        if not p.exists() or not p.is_file():
            print(f"\n❌ Error: File not found at '{cleaned_path}'. Please check the path.")
            return False

        if p.suffix.lower() not in [".pdf", ".md", ".txt"]:
            print(f"\n❌ Error: Unsupported file type '{p.suffix}'. Supported: .pdf, .md, .txt")
            return False

        print(f"\n⏳ Parsing and indexing '{p.name}'...")
        try:
            new_chunks_count = self.rag_pipeline.add_document(p)
            print(f"✅ Successfully indexed '{p.name}' (+{new_chunks_count} chunks). Total corpus: {len(self.rag_pipeline.chunks)} chunks.")
            print(f"You can now ask questions about '{p.name}'!\n")
            return True
        except Exception as e:
            print(f"❌ Failed to parse '{p.name}': {e}\n")
            return False

    def run_turn(self, user_question: str) -> Dict[str, Any]:
        """Runs a single conversational RAG turn with guardrails, citations, and history."""
        start_time = time.perf_counter()

        # Hybrid Retrieval + Preprocessing + Reranking + Generation
        res = self.rag_pipeline.query(
            question=user_question,
            top_k=4,
            conversation_history=self.conversation_history[1:]  # skip system prompt in turn history
        )

        final_answer = res["answer"]
        is_refusal = res["is_refusal"]
        citations = res.get("citations", [])
        top_confidence = res["top_confidence"]
        provider_name = res["provider_used"]
        latency_ms = res["total_latency_ms"]

        # Store in message history dictionary for this active session
        self.conversation_history.append({"role": "user", "content": user_question})
        self.conversation_history.append({"role": "assistant", "content": final_answer})

        return {
            "answer": final_answer,
            "is_refusal": is_refusal,
            "citations": citations,
            "confidence": top_confidence,
            "latency_ms": latency_ms,
            "provider": provider_name
        }

    def print_history(self):
        """Prints the full conversation message dictionary."""
        print("\n" + "=" * 65)
        print(f"Message History Dictionary ({len(self.conversation_history)} messages):")
        print("=" * 65)
        for idx, msg in enumerate(self.conversation_history):
            role_badge = f"[{msg['role'].upper()}]"
            content_preview = msg["content"]
            if len(content_preview) > 160 and idx > 0:
                content_preview = content_preview[:160] + "..."
            print(f"{idx}. {role_badge}: {content_preview}")
        print("=" * 65 + "\n")

    def reset_history(self):
        """Resets the history dictionary back to system prompt."""
        self.conversation_history = [
            {"role": "system", "content": self.system_prompt}
        ]
        print("\nConversation history cleared. System prompt preserved.\n")


def main():
    parser = argparse.ArgumentParser(description="AlphaResearch Terminal Financial & PDF RAG")
    parser.add_argument("--pdf", type=str, default="", help="Optional path to a local PDF to index immediately")
    args = parser.parse_args()

    engine = InteractiveFinancialChatPipeline()

    # If user provided a PDF at startup
    if args.pdf:
        engine.load_custom_pdf(args.pdf)

    print("\n" + "=" * 65)
    print("💬 AlphaResearch Interactive Terminal Chat")
    print("Tips:")
    print("  * Ask questions about Apple, Microsoft, Tesla, Alphabet, Amazon")
    print("  * Drag & drop any local PDF file path directly into the prompt to index it!")
    print("  * Type 'load <path_to_pdf>' to index a new document")
    print("  * Type 'history' to inspect the conversation message dictionary")
    print("  * Type 'clear' to reset memory | 'exit' to quit")
    print("=" * 65 + "\n")

    while True:
        try:
            user_input = input("You > ").strip()
            if not user_input:
                continue

            cleaned_input = user_input.strip("'\"")

            # Check if input is a dragged-and-dropped PDF/file path
            if (cleaned_input.lower().endswith(".pdf") or
                cleaned_input.lower().endswith(".md") or
                cleaned_input.lower().endswith(".txt")) and Path(cleaned_input).exists():
                engine.load_custom_pdf(cleaned_input)
                continue

            # Command checks
            cmd = user_input.lower()
            if cmd in ["exit", "quit", "q"]:
                print("\nExiting Financial RAG Assistant. Goodbye!")
                break
            elif cmd == "history":
                engine.print_history()
                continue
            elif cmd in ["clear", "new"]:
                engine.reset_history()
                continue
            elif cmd == "prompt":
                print(f"\nCurrent System Prompt:\n{engine.system_prompt}\n")
                continue
            elif cmd.startswith("load "):
                file_target = user_input[5:].strip()
                engine.load_custom_pdf(file_target)
                continue

            # Execute RAG Turn
            result = engine.run_turn(user_input)

            # Output
            print("\nAssistant >")
            print(result["answer"])

            if result["citations"]:
                print("\nSources:")
                for cite in result["citations"][:3]:
                    print(f"  * {cite}")

            print(f"\n[Confidence: {result['confidence']:.2f} | Latency: {result['latency_ms']:.1f}ms | Backend: {result['provider']}]")
            print("-" * 65 + "\n")

        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting.")
            break
        except Exception as e:
            print(f"\nError: {e}\n")

if __name__ == "__main__":
    main()
