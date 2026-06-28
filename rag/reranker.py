"""Cross-encoder reranker using BAAI/bge-reranker-base.

Replaces Cohere reranker with a local cross-encoder model.
Scores each query-document pair independently for fine-grained
relevance ranking after the initial retrieval stage.
"""

from sentence_transformers import CrossEncoder
from config import RERANKER_MODEL

_reranker = None


def get_reranker():
    """Return a singleton CrossEncoder instance."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def rerank(query, documents, top_k=5):
    """Rerank documents using the cross-encoder model.

    Unlike bi-encoders (used for initial retrieval), cross-encoders
    process the query and document together, producing more accurate
    relevance scores at the cost of speed.

    Args:
        query: The user's question.
        documents: List of LangChain Document objects.
        top_k: Number of top results to return.

    Returns:
        List of (Document, score) tuples, sorted by relevance descending.
        Scores are raw logits from the cross-encoder.
    """
    if not documents:
        return []

    reranker = get_reranker()
    pairs = [(query, doc.page_content) for doc in documents]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return [(doc, float(score)) for doc, score in ranked[:top_k]]
