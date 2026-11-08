"""
==============================================================================
AlphaResearch | AI Financial Analyst - Streamlit Application
==============================================================================
Clean single-page conversational interface:
- Document uploader placed directly in the chat interface
- Session-isolated document uploads (uploaded files are visible ONLY to you)
- Groq API key box removed from sidebar (loaded securely in background)
- Sidebar dedicated exclusively to your session's uploaded files & New Chat
==============================================================================
"""

import streamlit as st
import time
from pathlib import Path
import os
import numpy as np

from src.config import config, REPORTS_DIR
from src.pipeline import FinancialRAGPipeline
from src.vector_store import FinancialVectorStore
from src.lexical_search import BM25OkapiRetriever
from src.hybrid_retriever import HybridFinancialRetriever

# Page Setup
st.set_page_config(
    page_title="AlphaResearch | AI Financial Analyst",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
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
    .upload-card {
        background-color: #f8f9fa;
        border: 1px dashed #ced4da;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)


# 1. Base Corporate Pipeline (Pre-indexed once & shared globally across users)
@st.cache_resource(show_spinner="Loading Corporate 10-K Filings & AI Models...")
def get_base_pipeline():
    pipe = FinancialRAGPipeline(
        use_hybrid=True,
        use_reranker=True,
        prefer_neural=True,
        llm_provider="groq"
    )
    pipe.load_and_index(REPORTS_DIR)
    return pipe

base_pipeline = get_base_pipeline()


# 2. Session-Specific Pipeline Factory (Private & Isolated to each user's browser session)
def create_session_pipeline(base: FinancialRAGPipeline) -> FinancialRAGPipeline:
    """Clones pipeline state so any uploaded files are 100% private to this session."""
    session_pipe = FinancialRAGPipeline.__new__(FinancialRAGPipeline)
    session_pipe.use_hybrid = True
    session_pipe.use_reranker = True
    session_pipe.prefer_neural = True
    session_pipe.ingestion_engine = base.ingestion_engine
    session_pipe.chunker = base.chunker
    session_pipe.embedding_engine = base.embedding_engine
    session_pipe.reranker = base.reranker
    session_pipe.llm_client = base.llm_client
    session_pipe.query_preprocessor = base.query_preprocessor

    # Copy base document and chunk references
    session_pipe.documents = list(base.documents)
    session_pipe.chunks = list(base.chunks)

    # Isolated vector store for this user session
    session_pipe.vector_store = FinancialVectorStore()
    session_pipe.vector_store.chunks = list(base.vector_store.chunks)
    session_pipe.vector_store.embeddings = np.copy(base.vector_store.embeddings)

    # Isolated BM25 index for this user session
    session_pipe.bm25_retriever = BM25OkapiRetriever()
    session_pipe.bm25_retriever.index_chunks(session_pipe.chunks)

    session_pipe.hybrid_retriever = HybridFinancialRetriever(
        session_pipe.vector_store,
        session_pipe.bm25_retriever,
        session_pipe.embedding_engine
    )
    session_pipe.is_indexed = True
    return session_pipe


# Initialize Session State
if "pipeline" not in st.session_state:
    st.session_state.pipeline = create_session_pipeline(base_pipeline)

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "uploaded_files_info" not in st.session_state:
    st.session_state.uploaded_files_info = []

pipeline: FinancialRAGPipeline = st.session_state.pipeline


# ----------------------------------------------------------------------------
# Sidebar: Only for Session Controls & Uploaded Files
# ----------------------------------------------------------------------------
with st.sidebar:
    st.title("📈 AlphaResearch")
    st.caption("AI Financial Analyst")

    # Start New Chat Button
    if st.button("🔄 Start New Chat", use_container_width=True, type="primary"):
        st.session_state.chat_messages = []
        st.session_state.pipeline = create_session_pipeline(base_pipeline)
        st.session_state.uploaded_files_info = []
        st.rerun()

    st.markdown("---")

    # Session Uploaded Files Section (Private to this user only)
    st.subheader("📎 Your Uploaded Files")
    st.caption("🔒 Private to your session: only you can view and search your uploaded files.")

    if st.session_state.uploaded_files_info:
        for idx, uinfo in enumerate(st.session_state.uploaded_files_info, start=1):
            st.markdown(f"**{idx}. {uinfo['name']}**")
            st.caption(f"Added {uinfo['chunks']} chunks | Size: {uinfo['size_kb']:.1f} KB")

        st.markdown("")
        if st.button("🗑️ Clear My Uploaded Files", use_container_width=True):
            st.session_state.pipeline = create_session_pipeline(base_pipeline)
            st.session_state.uploaded_files_info = []
            st.success("Uploaded files removed. Reverted to corporate filings.")
            time.sleep(1)
            st.rerun()
    else:
        st.info("No files uploaded yet in this session. You can upload any PDF or document directly in the chat!")

    st.markdown("---")
    st.subheader("🏢 Corporate Knowledge Base")
    st.caption("Pre-indexed filings available to all queries:")
    st.caption("• Apple Inc. (`AAPL`)")
    st.caption("• Microsoft Corporation (`MSFT`)")
    st.caption("• Tesla, Inc. (`TSLA`)")
    st.caption("• Alphabet Inc. (`GOOGL`)")
    st.caption("• Amazon.com, Inc. (`AMZN`)")


# ----------------------------------------------------------------------------
# Main Page: Header, Direct Upload Box & Chat Stream
# ----------------------------------------------------------------------------
header_left, header_right = st.columns([5, 1])
with header_left:
    st.title("Financial Intelligence Assistant")
    st.markdown("Query verified corporate SEC 10-K filings or upload your own financial documents to analyze them live.")
with header_right:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_messages = []
        st.rerun()

# 📤 Document Uploader Directly in Chat Interface
with st.expander("📤 Upload Custom Document / PDF (Optional)", expanded=len(st.session_state.uploaded_files_info) == 0):
    uploaded_file = st.file_uploader(
        "Drag and drop any PDF, Markdown, or text filing here:",
        type=["pdf", "md", "txt"],
        key="main_chat_uploader",
        help="Files uploaded here are stored in memory for your session only. They will not be visible to any other user."
    )
    if uploaded_file is not None:
        # Check if already uploaded in this session
        already_indexed = any(u["name"] == uploaded_file.name for u in st.session_state.uploaded_files_info)
        if not already_indexed:
            with st.spinner(f"Extracting and indexing '{uploaded_file.name}' for your session..."):
                # Save into a session-specific scratch file
                scratch_dir = Path("data") / "uploads" / "session"
                scratch_dir.mkdir(parents=True, exist_ok=True)
                temp_path = scratch_dir / f"session_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                new_chunks = pipeline.add_document(temp_path)
                st.session_state.uploaded_files_info.append({
                    "name": uploaded_file.name,
                    "chunks": new_chunks,
                    "size_kb": len(uploaded_file.getbuffer()) / 1024.0
                })

                # Remove temporary disk file immediately to preserve privacy
                try:
                    temp_path.unlink()
                except Exception:
                    pass

                st.success(f"✅ Successfully indexed '{uploaded_file.name}' (+{new_chunks} chunks)! It is now private to your chat.")
                time.sleep(1)
                st.rerun()

# Quick Prompt Suggestions (Chips)
st.caption("Quick sample inquiries:")
q1, q2, q3, q4 = st.columns(4)
suggested_query = None

with q1:
    if st.button("Apple Net Sales & iPhone", use_container_width=True):
        suggested_query = "What was Apple's total net sales and iPhone revenue in fiscal 2023?"
with q2:
    if st.button("Microsoft Cloud Revenue", use_container_width=True):
        suggested_query = "How much revenue did Microsoft Intelligent Cloud produce in FY23?"
with q3:
    if st.button("Tesla Gross Margin", use_container_width=True):
        suggested_query = "What was Tesla's gross margin and why did operating income decline?"
with q4:
    if st.button("Amazon AWS Operating Profit", use_container_width=True):
        suggested_query = "What portion of Amazon's total operating profit came from AWS in 2023?"

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

# Chat Input Box at Bottom
user_query = st.chat_input("Ask a question about financial disclosures or your uploaded document...")
active_inquiry = user_query or suggested_query

if active_inquiry:
    # 1. Add User Message
    st.session_state.chat_messages.append({"role": "user", "content": active_inquiry})
    with st.chat_message("user"):
        st.markdown(active_inquiry)

    # 2. Run Query through this Session's Private Pipeline
    with st.chat_message("assistant"):
        with st.spinner("Analyzing verified documents..."):
            t_start = time.perf_counter()
            result = pipeline.query(
                active_inquiry,
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

        if suggested_query:
            st.rerun()
