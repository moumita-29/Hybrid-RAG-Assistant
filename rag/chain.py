"""RAG chain — retrieves context and generates answers with source citations.

Combines the hybrid retriever with the LLM to answer
questions grounded in the uploaded documents.
Phase 4: Added conversation memory and retrieval logging.
"""

import os
import time
import logging
from rag.retriever import hybrid_search
from rag.llm import get_llm
from rag.confidence import compute_confidence
from config import TOP_K

logger = logging.getLogger(__name__)

# Reused from the original project
RAG_SYSTEM_PROMPT = """
You are a friendly and knowledgeable assistant that provides complete and insightful answers.
Answer the user's question using only the context below.
When responding, you MUST NOT reference the existence of the context, directly or indirectly.
Instead, you MUST treat the context as if its contents are entirely part of your working memory.
""".strip()

FALLBACK_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. When you don't know something, "
    "be honest about it. Provide clear, concise, and accurate responses."
)


# ---------- Retrieval ----------


def retrieve(query, vector_store, bm25_index=None, k=TOP_K):
    """Retrieve relevant documents using hybrid search + reranking."""
    start_time = time.time()
    
    if vector_store is None:
        logger.info(f"Retrieval skipped: No vector store available for query '{query}'")
        return [], [], (0.0, "Low", "🔴"), {}

    results = hybrid_search(query, vector_store, bm25_index, k=k)

    if not results:
        logger.info(f"Retrieval returned 0 results for query '{query}'")
        return [], [], (0.0, "Low", "🔴"), {}

    context_parts = []
    sources = []
    for doc, score in results:
        context_parts.append(doc.page_content)
        page = doc.metadata.get("page", None)
        sources.append({
            "content": (doc.page_content[:200] + "...")
            if len(doc.page_content) > 200
            else doc.page_content,
            "source": doc.metadata.get("source", "Unknown"),
            "page": page + 1 if page is not None else "N/A",
        })

    confidence = compute_confidence(results, score_type="reranker")
    
    elapsed = time.time() - start_time
    logger.info(
        f"Retrieval complete in {elapsed:.2f}s | "
        f"Found {len(results)} chunks | "
        f"Confidence: {confidence[0]}% ({confidence[1]}) | "
        f"Query: '{query}'"
    )
    
    debug_info = {
        "query": query,
        "time_taken_sec": elapsed,
        "num_chunks_retrieved": len(results),
        "confidence_score": confidence[0],
        "chunks": [
            {
                "score": float(score),
                "source": os.path.basename(doc.metadata.get("source", "Unknown")).replace("temp_", ""),
                "page": doc.metadata.get("page", 0) + 1,
                "content": doc.page_content[:300] + "..."
            }
            for doc, score in results
        ]
    }
    
    return context_parts, sources, confidence, debug_info


# ---------- Generation (with Memory) ----------


def _build_rag_prompt(query, context_parts):
    """Build the RAG prompt from query and context chunks."""
    context = "\n\n---\n\n".join(context_parts)
    return (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Provide a comprehensive answer based on the context above. "
        f"If the context doesn't contain enough information, say so clearly."
    )


def _build_messages(system_prompt, query, context_parts=None, chat_history=None):
    """Construct message list including system prompt, history, and current query."""
    messages = [("system", system_prompt)]
    
    if chat_history:
        for turn in chat_history:
            messages.append(("human", turn["question"]))
            messages.append(("ai", turn["answer"]))
            
    if context_parts:
        prompt = _build_rag_prompt(query, context_parts)
    else:
        prompt = query
        
    messages.append(("human", prompt))
    return messages


def generate_stream(query, context_parts, chat_history=None):
    """Stream LLM response tokens for a RAG query, using chat history."""
    llm = get_llm()
    messages = _build_messages(RAG_SYSTEM_PROMPT, query, context_parts, chat_history)
    
    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content


def fallback_stream(query, chat_history=None):
    """Stream a fallback response when no documents match, using chat history."""
    llm = get_llm()
    messages = _build_messages(FALLBACK_SYSTEM_PROMPT, query, None, chat_history)
    
    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content


# ---------- Non-streaming (kept for compatibility/testing) ----------


def ask(query, vector_store, bm25_index=None, k=TOP_K, chat_history=None):
    """Non-streaming RAG query."""
    context_parts, sources, confidence, debug_info = retrieve(query, vector_store, bm25_index, k)

    if not context_parts:
        return _fallback(query, chat_history), [], (0.0, "Low", "🔴"), {}

    llm = get_llm()
    messages = _build_messages(RAG_SYSTEM_PROMPT, query, context_parts, chat_history)
    response = llm.invoke(messages)
    return response.content, sources, confidence, debug_info


def _fallback(query, chat_history=None):
    """Answer using general knowledge when no documents match."""
    llm = get_llm()
    messages = _build_messages(FALLBACK_SYSTEM_PROMPT, query, None, chat_history)
    response = llm.invoke(messages)
    return response.content
