"""
==============================================================================
AlphaResearch | AI Financial Analyst - Streamlit Application
==============================================================================
A clean, single-page conversational RAG interface:
- Interactive multi-turn chat stream
- One-click "New Chat" session reset
- Drag-and-drop PDF / Document uploader
- Grounded citations & confidence guardrails
==============================================================================
"""

import streamlit as st
import time
from pathlib import Path
import os

from src.config import config, REPORTS_DIR
from src.pipeline import FinancialRAGPipeline

# Page Configuration
st.set_page_config(
    page_title="AlphaResearch | AI Financial Analyst",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Clean Styling
st.markdown("""
<style>
    .citation-tag {
        display: inline-block;
        background-color: #e9ecef;
        color: #495057;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
        margin-right: 6px;
        margin-bottom: 4px;
        border: 1px solid #ced4da;
    }
    .refusal-box {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeeba;
        padding: 12px;
        border-radius: 6px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Cache Pipeline Instance
@st.cache_resource(show_spinner="Initializing AI Financial Analyst & Documents...")
def get_pipeline():
    pipe = FinancialRAGPipeline(
        use_hybrid=True,
        use_reranker=True,
        prefer_neural=True,
        llm_provider="groq"
    )
    pipe.load_and_index(REPORTS_DIR)
    return pipe

pipeline = get_pipeline()

# Session State for Active Chat History
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# ----------------------------------------------------------------------------
# Sidebar: Controls, Uploads & Settings
# ----------------------------------------------------------------------------
with st.sidebar:
    st.title("📈 AlphaResearch")
    st.caption("AI Financial Analyst Assistant")

    # 1. New Chat Button
    if st.button("🔄 Start New Chat", use_container_width=True, type="primary"):
        st.session_state.chat_messages = []
        st.rerun()

    st.markdown("---")

    # 2. Upload Document
    st.subheader("📤 Upload Document")
    uploaded_file = st.file_uploader(
        "Drag & drop PDF or text report",
        type=["pdf", "md", "txt"],
        help="Upload any financial filing or document to query it live."
    )
    if uploaded_file is not None:
        save_path = REPORTS_DIR / uploaded_file.name
        if not save_path.exists():
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            added_chunks = pipeline.add_document(save_path)
            st.success(f"✅ Indexed '{uploaded_file.name}' (+{added_chunks} chunks)!")
            time.sleep(1)
            st.rerun()
        else:
            st.info(f"'{uploaded_file.name}' is indexed and ready.")

    st.markdown("---")

    # 3. LLM API Key Settings
    st.subheader("🔑 Groq API Key")
    default_groq = os.getenv("GROQ_API_KEY", "")
    if not default_groq and hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
        default_groq = st.secrets["GROQ_API_KEY"]

    api_key_input = st.text_input(
        "API Key (Free at console.groq.com)",
        type="password",
        value=default_groq,
        placeholder="Paste gsk_... key here"
    )
    if api_key_input:
        pipeline.llm_client.set_keys(groq_key=api_key_input)
        pipeline.llm_client.provider = "groq"
    else:
        pipeline.llm_client.provider = "offline"

    st.markdown("---")

    # 4. Indexed Documents List
    st.subheader("📚 Available Documents")
    for doc in pipeline.documents:
        st.caption(f"• **{doc.company}** ({doc.ticker})")

# ----------------------------------------------------------------------------
# Main Page: Conversational Chat Interface
# ----------------------------------------------------------------------------
header_col1, header_col2 = st.columns([5, 1])
with header_col1:
    st.title("Financial Intelligence Assistant")
    st.markdown("Ask quantitative or analytical questions about SEC 10-K filings and your uploaded documents.")
with header_col2:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_messages = []
        st.rerun()

# Quick Prompt Suggestions (Chips)
st.caption("Quick suggestions to try:")
quick_col1, quick_col2, quick_col3, quick_col4 = st.columns(4)
suggested_prompt = None

with quick_col1:
    if st.button("Apple Net Sales & iPhone", use_container_width=True):
        suggested_prompt = "What was Apple's total net sales and iPhone revenue in fiscal 2023?"
with quick_col2:
    if st.button("Microsoft Cloud Revenue", use_container_width=True):
        suggested_prompt = "How much revenue did Microsoft Intelligent Cloud produce in FY23?"
with quick_col3:
    if st.button("Tesla Gross Margin", use_container_width=True):
        suggested_prompt = "What was Tesla's gross margin and why did operating income decline?"
with quick_col4:
    if st.button("Amazon AWS Profit", use_container_width=True):
        suggested_prompt = "What portion of Amazon's total operating profit came from AWS in 2023?"

st.markdown("---")

# Render Conversation Stream
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        if msg.get("is_refusal"):
            st.markdown(f"<div class='refusal-box'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(msg["content"])

        if msg.get("citations"):
            citation_html = "".join([f"<span class='citation-tag'>📄 {c}</span>" for c in msg["citations"]])
            st.markdown(citation_html, unsafe_allow_html=True)

        if "confidence" in msg and msg.get("confidence") is not None:
            st.caption(
                f"Confidence: **{msg['confidence']:.2f}** | "
                f"Latency: **{msg.get('latency_ms', 0):.1f} ms** | "
                f"Engine: `{msg.get('provider', '')}`"
            )

# Accept User Input
user_input = st.chat_input("Ask a question about financial disclosures or your uploaded file...")
active_query = user_input or suggested_prompt

if active_query:
    # 1. Append User Message
    st.session_state.chat_messages.append({"role": "user", "content": active_query})
    with st.chat_message("user"):
        st.markdown(active_query)

    # 2. Run Query through RAG Pipeline
    with st.chat_message("assistant"):
        with st.spinner("Analyzing verified documents..."):
            t_start = time.perf_counter()
            result = pipeline.query(
                active_query,
                top_k=3,
                conversation_history=st.session_state.chat_messages[:-1]
            )
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        if result["is_refusal"]:
            st.markdown(f"<div class='refusal-box'>{result['answer']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(result["answer"])

        if result.get("citations"):
            citation_html = "".join([f"<span class='citation-tag'>📄 {c}</span>" for c in result["citations"]])
            st.markdown(citation_html, unsafe_allow_html=True)

        st.caption(
            f"Confidence: **{result['top_confidence']:.2f}** | "
            f"Latency: **{elapsed_ms:.1f} ms** | "
            f"Engine: `{result['provider_used']}`"
        )

        # Store in session state
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": result["answer"],
            "is_refusal": result["is_refusal"],
            "citations": result.get("citations", []),
            "confidence": result["top_confidence"],
            "latency_ms": elapsed_ms,
            "provider": result["provider_used"]
        })

        if suggested_prompt:
            st.rerun()
