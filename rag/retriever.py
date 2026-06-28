"""Hybrid retriever: FAISS + BM25 → Reciprocal Rank Fusion → Reranking.

This is the core of the hybrid search pipeline:
1. FAISS (semantic search) — finds documents with similar meaning
2. BM25 (keyword search)  — finds documents with matching terms
3. Reciprocal Rank Fusion — merges both ranked lists fairly
4. Cross-encoder reranker — rescores the merged candidates
"""

from rag.vector_store import similarity_search
from rag.reranker import rerank
from config import TOP_K, BM25_K, RRF_K


def hybrid_search(query, vector_store, bm25_index, k=TOP_K):
    """Perform hybrid search combining semantic and keyword retrieval.

    Args:
        query: The user's question.
        vector_store: FAISS vector store instance.
        bm25_index: BM25Index instance (can be None for FAISS-only).
        k: Number of final results to return after reranking.

    Returns:
        List of (Document, reranker_score) tuples.
    """
    # Step 1: Get candidates from both retrievers
    faiss_results = similarity_search(vector_store, query, k=BM25_K)
    bm25_results = bm25_index.search(query, k=BM25_K) if bm25_index else []

    # Step 2: Merge with Reciprocal Rank Fusion
    fused = reciprocal_rank_fusion(faiss_results, bm25_results, k=RRF_K)

    if not fused:
        return []

    # Step 3: Rerank the top fused candidates with cross-encoder
    # Slicing to top 8 candidates drastically speeds up CPU inference
    fused_docs = [doc for doc, _ in fused[:8]]
    return rerank(query, fused_docs, top_k=k)


def reciprocal_rank_fusion(faiss_results, bm25_results, k=60):
    """Merge two ranked result lists using Reciprocal Rank Fusion (RRF).

    RRF score = Σ  1 / (k + rank)  for each list the document appears in.

    This balances semantic and keyword relevance without needing to
    normalize scores across the two very different retrieval methods.
    A document ranked highly by both methods gets a higher fused score.

    Args:
        faiss_results: List of (Document, score) from FAISS.
        bm25_results: List of (Document, score) from BM25.
        k: RRF constant (default 60). Higher values reduce the
           impact of high rankings from a single source.

    Returns:
        List of (Document, rrf_score) tuples sorted by fused score.
    """
    doc_scores = {}  # content_hash → accumulated RRF score
    doc_map = {}     # content_hash → Document object

    for rank, (doc, _) in enumerate(faiss_results):
        key = hash(doc.page_content)
        doc_scores[key] = doc_scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        doc_map[key] = doc

    for rank, (doc, _) in enumerate(bm25_results):
        key = hash(doc.page_content)
        doc_scores[key] = doc_scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        doc_map[key] = doc

    sorted_results = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_map[key], score) for key, score in sorted_results]
