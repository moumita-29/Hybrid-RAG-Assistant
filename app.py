"""Streamlit UI for the Hybrid Search RAG Assistant.

Phase 3: Integrated BM25 for true Hybrid Search, along with
Cross-encoder reranking.

No API keys required — everything runs locally.
"""

import os
import json
import logging
import urllib.request
import streamlit as st
import warnings

from rag.loader import load_and_chunk_pdf
from rag.vector_store import build_index, load_index, add_documents
from rag.chain import retrieve, generate_stream, fallback_stream
from rag.bm25 import BM25Index

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message=".*torch.classes.*")


# ---------- Helper functions ----------


from dotenv import load_dotenv
load_dotenv()


def process_document(file_path: str) -> bool:
    """Processes a PDF document: loads, chunks, and adds to FAISS/BM25 indices.

    Reuses the original project's processing pattern but replaces
    raglite's insert_document with our loader + FAISS/BM25 pipeline.
    """
    try:
        chunks = load_and_chunk_pdf(file_path)
        if not chunks:
            return False

        if st.session_state.get("vector_store") is None:
            st.session_state.vector_store = build_index(chunks)
            st.session_state.bm25_index = BM25Index(chunks)
        else:
            st.session_state.vector_store = add_documents(
                st.session_state.vector_store, chunks
            )
            st.session_state.bm25_index.add_documents(chunks)
        return True
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        return False


def display_sources(sources):
    """Render source citations in an expandable section."""
    if not sources:
        return
    with st.expander("📄 Sources"):
        for i, src in enumerate(sources, 1):
            source_name = os.path.basename(str(src["source"])).replace("temp_", "")
            st.markdown(f"**Source {i}** — {source_name}, Page {src['page']}")
            st.caption(src["content"])
            if i < len(sources):
                st.markdown("---")


def display_source_badge(is_fallback, confidence=None):
    """Show a badge indicating whether the answer used RAG or General Knowledge."""
    if is_fallback:
        st.caption("🟡 **Answer Generated from General Knowledge**")
    else:
        score = confidence[0] if confidence else 0.0
        st.caption(f"🟢 **Answer Based on Uploaded Documents** (Confidence: {score:.1f}%)")


# ---------- Main app ----------


def main():
    st.set_page_config(page_title="Hybrid Search RAG Assistant", layout="wide")

    # --- Session state initialization ---
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = load_index()
        
    if "bm25_index" not in st.session_state:
        st.session_state.bm25_index = None
        if st.session_state.vector_store:
            # Rebuild BM25 index from FAISS docstore
            docs = list(st.session_state.vector_store.docstore._dict.values())
            st.session_state.bm25_index = BM25Index(docs)

    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = st.session_state.vector_store is not None

    if "metrics" not in st.session_state:
        st.session_state.metrics = []
    if "last_debug_info" not in st.session_state:
        st.session_state.last_debug_info = None

    # --- Sidebar ---
    with st.sidebar:
        st.title("⚙️ Configuration")

        # HF API check
        from config import HF_LLM_MODEL
        if os.environ.get("HF_TOKEN"):
            st.success("✅ HF_TOKEN is configured")
        else:
            st.error("❌ HF_TOKEN is missing in .env")

        st.markdown("---")
        st.markdown("**Models**")
        st.info(f"🤖 LLM: {HF_LLM_MODEL.split('/')[-1]} (API)")
        
        from config import EMBEDDING_MODEL, RERANKER_MODEL
        st.info(f"📐 Embeddings: {EMBEDDING_MODEL}")
        st.info(f"🎯 Reranker: {RERANKER_MODEL.split('/')[-1]}")

        st.markdown("---")
        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.session_state.metrics = []
                st.session_state.last_debug_info = None
                st.rerun()

    # --- Main area ---
    st.title("👀 RAG App with Hybrid Search")

    tab1, tab2, tab3 = st.tabs(["💬 Chat", "🔍 Retrieval Inspector", "📊 Evaluation Dashboard"])

    with tab1:
        # File uploader — reuses the original multi-file upload pattern
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_uploader",
        )

        if uploaded_files:
            success = False
            for uploaded_file in uploaded_files:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getvalue())

                    if process_document(temp_path):
                        st.success(f"✅ Processed: {uploaded_file.name}")
                        success = True
                    else:
                        st.error(f"❌ Failed: {uploaded_file.name}")
                    os.remove(temp_path)

            if success:
                st.session_state.documents_loaded = True

        # --- Chat interface ---
        if st.session_state.documents_loaded:
            # Replay chat history
            for msg in st.session_state.chat_history:
                with st.chat_message("user"):
                    st.write(msg["question"])
                with st.chat_message("assistant"):
                    display_source_badge(msg.get("is_fallback", False), msg.get("confidence"))
                    st.write(msg["answer"])
                    display_sources(msg.get("sources", []))

            # New question
            user_input = st.chat_input("Ask a question about your documents...")
            if user_input:
                with st.chat_message("user"):
                    st.write(user_input)

                with st.chat_message("assistant"):
                    # Step 1: Retrieve relevant chunks (now using hybrid search)
                    with st.spinner("Searching documents (may download AI models on first run)..."):
                        context_parts, sources, confidence, debug_info = retrieve(
                            user_input, 
                            st.session_state.vector_store,
                            st.session_state.bm25_index
                        )
                        st.session_state.last_debug_info = debug_info
                        if debug_info:
                            st.session_state.metrics.append(debug_info)

                    # Get recent history
                    from config import MEMORY_WINDOW, FALLBACK_THRESHOLD
                    recent_history = st.session_state.chat_history[-MEMORY_WINDOW:] if st.session_state.chat_history else None

                    # Step 2: Determine Fallback
                    conf_score = confidence[0] if confidence else 0.0
                    conf_label = confidence[1] if confidence else "Low"
                    
                    is_fallback = False
                    if not context_parts or conf_score < FALLBACK_THRESHOLD or conf_label == "Low":
                        is_fallback = True

                    display_source_badge(is_fallback, confidence)

                    if is_fallback:
                        logger.info(f"Fallback triggered for query: '{user_input}'. Score: {conf_score}")
                        st.info("No relevant information was found in the uploaded documents. The following answer is generated using the model's general knowledge.")
                        answer = st.write_stream(fallback_stream(user_input, recent_history))
                        sources = []  # Clear sources to avoid fake citations
                    else:
                        logger.info(f"RAG triggered for query: '{user_input}'. Score: {conf_score}")
                        answer = st.write_stream(
                            generate_stream(user_input, context_parts, recent_history)
                        )

                    # Step 4: Show sources below the answer
                    display_sources(sources)

                    # Step 5: Save to history
                    st.session_state.chat_history.append({
                        "question": user_input,
                        "answer": answer,
                        "sources": sources,
                        "confidence": confidence,
                        "is_fallback": is_fallback,
                    })
        else:
            st.info("📄 Upload PDF documents to get started.")

    with tab2:
        st.header("🔍 Retrieval Inspector")
        st.markdown("Examine the exact chunks retrieved by the Hybrid Search engine for your latest query.")
        
        if st.session_state.last_debug_info:
            debug = st.session_state.last_debug_info
            if not debug.get("chunks"):
                st.info("No chunks were retrieved for the last query.")
            else:
                st.metric("Time Taken (s)", f"{debug.get('time_taken_sec', 0):.2f}")
                for i, chunk in enumerate(debug["chunks"], 1):
                    with st.expander(f"Rank {i} | Score: {chunk['score']:.4f} | {chunk['source']} (Page {chunk['page']})"):
                        st.markdown(chunk["content"])
        else:
            st.info("Ask a question in the chat to see retrieval diagnostics.")

    with tab3:
        st.header("📊 RAG Evaluation Dashboard")
        st.markdown("Overall metrics and performance stats for your current session.")
        
        if not st.session_state.metrics:
            st.info("No data yet. Start chatting to populate the dashboard!")
        else:
            metrics = st.session_state.metrics
            successful_queries = [m for m in metrics if m.get("chunks")]
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Queries", len(metrics))
            
            if successful_queries:
                avg_time = sum(m.get("time_taken_sec", 0) for m in successful_queries) / len(successful_queries)
                col2.metric("Avg Retrieval Time", f"{avg_time:.2f} s")
                
                avg_conf = sum(m.get("confidence_score", 0) for m in successful_queries) / len(successful_queries)
                col3.metric("Avg Confidence", f"{avg_conf:.1f}%")
                
                st.subheader("Query History")
                for i, m in enumerate(reversed(metrics), 1):
                    with st.expander(f"Q: {m['query']} ({m.get('time_taken_sec',0):.2f}s)"):
                        st.write(f"Chunks Retrieved: {m.get('num_chunks_retrieved', 0)}")
                        st.write(f"Confidence: {m.get('confidence_score', 0):.1f}%")


if __name__ == "__main__":
    main()