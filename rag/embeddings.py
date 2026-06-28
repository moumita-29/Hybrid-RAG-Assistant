"""Embedding model wrapper using BAAI/bge-base-en-v1.5.

Replaces OpenAI's text-embedding-3-large with a local
HuggingFace model. The model is loaded once and cached.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL

_embeddings = None


def get_embeddings():
    """Return a singleton HuggingFace embeddings instance.

    Uses BAAI/bge-base-en-v1.5 with normalized embeddings
    (required by the BGE model family for best results).
    """
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings
