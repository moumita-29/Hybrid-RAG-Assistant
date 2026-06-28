"""Confidence score calculation from retrieval scores.

Supports two score types:
- "l2": FAISS L2 distances (Phase 2, used as fallback)
- "reranker": Cross-encoder logits (Phase 3, used with hybrid search)
"""

import math


def compute_confidence(results, score_type="l2"):
    """Compute a confidence score from retrieval results.

    Args:
        results: List of (Document, score) tuples.
        score_type: "l2" for FAISS L2 distances,
                    "reranker" for cross-encoder logits.

    Returns:
        Tuple of (score, label, emoji):
        - score: Float 0-100 representing confidence percentage.
        - label: "High", "Medium", or "Low".
        - emoji: "🟢", "🟡", or "🔴".
    """
    if not results:
        return 0.0, "Low", "🔴"

    if score_type == "reranker":
        # Cross-encoder outputs raw logits → sigmoid → probability
        probs = [1.0 / (1.0 + math.exp(-score)) for _, score in results]
        avg_confidence = sum(probs) / len(probs) * 100
    else:
        # FAISS L2 distances → cosine similarity for normalized embeddings
        cos_sims = [max(0.0, 1.0 - (score ** 2) / 2.0) for _, score in results]
        avg_confidence = sum(cos_sims) / len(cos_sims) * 100

    if avg_confidence >= 70:
        return round(avg_confidence, 1), "High", "🟢"
    elif avg_confidence >= 40:
        return round(avg_confidence, 1), "Medium", "🟡"
    else:
        return round(avg_confidence, 1), "Low", "🔴"
