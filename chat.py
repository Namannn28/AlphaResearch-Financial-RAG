"""
==============================================================================
chat.py - Interactive Financial RAG Terminal Pipeline (No Streamlit Needed)
==============================================================================
Run locally in your terminal:
    python chat.py

Features:
- Pure CLI execution (lightweight, zero Streamlit overhead)
- Customizable System Prompt
- Multi-turn conversation history stored in a messages dictionary
- Hybrid RAG retrieval (BM25 + Dense) over SEC 10-K filings
- LangChain ChatGroq integration (Llama 3.3 70B via free GROQ_API_KEY)
- Out-of-corpus refusal guardrail & source citations
- Built-in commands: 'history', 'clear', 'prompt', 'exit'
==============================================================================
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# 1. Load environment variables (.env) and set UTF-8 console
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 2. Import RAG components
from src.config import config, REPORTS_DIR
from src.pipeline import FinancialRAGPipeline

# 3. LangChain Groq imports
try:
    from langchain_groq import ChatGroq
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False

# ----------------------------------------------------------------------------
# Customizable System Prompt
# ----------------------------------------------------------------------------
DEFAULT_SYSTEM_PROMPT = """You are an expert Senior Financial Analyst & Research Assistant specializing in SEC 10-K annual reports.
Your role is to assist analysts, investors, and recruiters by providing accurate, evidence-backed answers using verified 10-K filings.

OPERATIONAL RULES:
1. Base your answers strictly on the provided 10-K excerpt contexts and conversation history. Never speculate or hallucinate.
2. If the context does not contain enough information to answer, state:
   "[REFUSAL - INSUFFICIENT_EVIDENCE]: The provided 10-K filings do not contain documented evidence regarding this inquiry."
3. Always include citations: mention the company, ticker, and filing section (e.g., [Apple (AAPL) | Item 7 MD&A]).
4. Maintain a professional, quantitative, analytical tone with exact currency and percentage figures."""

class InteractiveFinancialChatPipeline:
    """Terminal-based interactive RAG pipeline with message history and system prompt."""

    def __init__(self, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        self.system_prompt = system_prompt
        self.groq_api_key = os.getenv("GROQ_API_KEY", "").strip()

        # Message History Dictionary (stores role and content for each turn)
        self.conversation_history: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]

        # Initialize Financial RAG Engine
        print("\n" + "=" * 65)
        print("Initializing AlphaResearch Financial RAG Engine...")
        print("=" * 65)
        self.rag_pipeline = FinancialRAGPipeline(
            use_section_chunker=True,
            use_hybrid=True,
            use_reranker=True,
            prefer_neural=True,
            llm_provider="groq"
        )
        self.rag_pipeline.load_and_index(REPORTS_DIR)
        print(f"Indexed {len(self.rag_pipeline.chunks)} section chunks across Apple, Microsoft, Tesla, Alphabet, and Amazon.")

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
                    print(f"Notice: ChatGroq initialization note ({e}). Will use offline fallback.")
        else:
            print("Notice: No GROQ_API_KEY detected in .env. Running in deterministic Offline Fallback mode.")

    def run_turn(self, user_question: str) -> Dict[str, Any]:
        """
        Executes a single conversational RAG turn:
        1. Retrieves relevant 10-K chunks using Hybrid search & Reranker.
        2. Applies confidence refusal guardrail.
        3. Formats prompt with verified context + conversation history.
        4. Calls LangChain ChatGroq.
        5. Records turn in message history dictionary.
        """
        start_time = time.perf_counter()

        # Step 1: Hybrid Retrieval + Cross-Encoder Reranking
        retrieval_res = self.rag_pipeline.query(user_question, top_k=3)
        top_confidence = retrieval_res["top_confidence"]
        is_refusal = retrieval_res["is_refusal"]
        chunks = retrieval_res["retrieved_chunks"]

        # Step 2: Guardrail Refusal Check
        if is_refusal:
            refusal_text = retrieval_res["answer"]
            # Store in history
            self.conversation_history.append({"role": "user", "content": user_question})
            self.conversation_history.append({"role": "assistant", "content": refusal_text})
            return {
                "answer": refusal_text,
                "is_refusal": True,
                "citations": [],
                "confidence": top_confidence,
                "latency_ms": (time.perf_counter() - start_time) * 1000.0,
                "provider": "GuardrailRefusal"
            }

        # Step 3: Format Context & Citations
        context_blocks = []
        citations = []
        for i, hit in enumerate(chunks, start=1):
            c = hit["chunk"]
            citation_str = f"{c.company} ({c.ticker}) | {c.section_title}"
            citations.append(citation_str)
            context_blocks.append(f"[Source {i}: {citation_str}]\n{c.text}")

        combined_context = "\n\n".join(context_blocks)

        # Build augmented user prompt with retrieved 10-Ks
        augmented_prompt = f"""Verified SEC 10-K Filing Contexts:
{combined_context}

User Question: {user_question}

Provide an accurate, concise answer with exact numbers and document citations:"""

        # Step 4: Generate Answer via LangChain ChatGroq (or offline fallback)
        provider_name = "Offline Grounded Engine"
        if self.chat_model is not None:
            try:
                # Convert message history dictionary to LangChain message objects
                langchain_messages = []
                langchain_messages.append(SystemMessage(content=self.system_prompt))

                # Include past conversational context (last 4 turns)
                for msg in self.conversation_history[1:][-4:]:
                    if msg["role"] == "user":
                        langchain_messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        langchain_messages.append(AIMessage(content=msg["content"]))

                # Add current turn with retrieved context
                langchain_messages.append(HumanMessage(content=augmented_prompt))

                response = self.chat_model.invoke(langchain_messages)
                final_answer = response.content.strip()
                provider_name = "LangChain (ChatGroq Llama 3.3 70B)"
            except Exception as e:
                final_answer = retrieval_res["answer"]
                provider_name = f"Offline Fallback (API error: {e})"
        else:
            final_answer = retrieval_res["answer"]

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Step 5: Update Message History Dictionary
        self.conversation_history.append({"role": "user", "content": user_question})
        self.conversation_history.append({"role": "assistant", "content": final_answer})

        return {
            "answer": final_answer,
            "is_refusal": False,
            "citations": citations,
            "confidence": top_confidence,
            "latency_ms": latency_ms,
            "provider": provider_name
        }

    def print_history(self):
        """Prints the internal message history dictionary."""
        print("\n" + "=" * 65)
        print(f"Conversation History Dictionary ({len(self.conversation_history)} messages):")
        print("=" * 65)
        for idx, msg in enumerate(self.conversation_history):
            role_badge = f"[{msg['role'].upper()}]"
            content_preview = msg["content"]
            if len(content_preview) > 180 and idx > 0:
                content_preview = content_preview[:180] + "..."
            print(f"{idx}. {role_badge}: {content_preview}")
        print("=" * 65 + "\n")

    def reset_history(self):
        """Resets the history dictionary back to just the system prompt."""
        self.conversation_history = [
            {"role": "system", "content": self.system_prompt}
        ]
        print("\nConversation history cleared. System prompt preserved.\n")


def start_interactive_cli():
    """Main CLI entry point for interactive terminal chat."""
    chat_engine = InteractiveFinancialChatPipeline()

    print("\n" + "=" * 65)
    print("📈 AlphaResearch Financial RAG - Interactive Terminal")
    print("Ask questions about Apple, Microsoft, Tesla, Alphabet, or Amazon 10-Ks.")
    print("Commands:")
    print("  'history' -> inspect conversation message dictionary")
    print("  'clear'   -> reset conversation memory")
    print("  'prompt'  -> view current system prompt")
    print("  'exit'    -> quit the chat")
    print("=" * 65 + "\n")

    while True:
        try:
            user_input = input("You > ").strip()
            if not user_input:
                continue

            # Command handling
            cmd_lower = user_input.lower()
            if cmd_lower in ["exit", "quit", "q"]:
                print("\nExiting Financial Research Assistant. Goodbye!")
                break
            elif cmd_lower == "history":
                chat_engine.print_history()
                continue
            elif cmd_lower == "clear":
                chat_engine.reset_history()
                continue
            elif cmd_lower == "prompt":
                print(f"\nCurrent System Prompt:\n{chat_engine.system_prompt}\n")
                continue

            # Execute RAG Turn
            result = chat_engine.run_turn(user_input)

            # Print Output
            print("\nAssistant >")
            print(result["answer"])

            if result["citations"]:
                print("\nSources:")
                for cite in result["citations"]:
                    print(f"  * {cite}")

            print(f"\n[Confidence: {result['confidence']:.2f} | Latency: {result['latency_ms']:.1f}ms | Backend: {result['provider']}]")
            print("-" * 65 + "\n")

        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting.")
            break
        except Exception as err:
            print(f"\nError processing query: {err}\n")

if __name__ == "__main__":
    start_interactive_cli()
