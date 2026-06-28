"""FAISS vector store management.

Replaces raglite's SQLite/PostgreSQL vector storage with
a local FAISS index. Supports build, load, add, and search.
"""

import os
from langchain_community.vectorstores import FAISS
from rag.embeddings import get_embeddings
from config import FAISS_INDEX_PATH, TOP_K


def build_index(documents):
    """Build a new FAISS index from documents and save to disk.

    Args:
        documents: List of LangChain Document objects.

    Returns:
        The FAISS vector store instance.
    """
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(FAISS_INDEX_PATH)
    return vector_store


def load_index():
    """Load a previously saved FAISS index from disk.

    Returns:
        The FAISS vector store instance, or None if no index exists.
    """
    if not os.path.exists(FAISS_INDEX_PATH):
        return None
    embeddings = get_embeddings()
    return FAISS.load_local(
        FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
    )


def add_documents(vector_store, documents):
    """Add new documents to an existing FAISS index and save.

    Args:
        vector_store: Existing FAISS vector store.
        documents: New LangChain Document objects to add.

    Returns:
        The updated FAISS vector store instance.
    """
    vector_store.add_documents(documents)
    vector_store.save_local(FAISS_INDEX_PATH)
    return vector_store


def similarity_search(vector_store, query, k=TOP_K):
    """Search the FAISS index for the most similar documents.

    Args:
        vector_store: FAISS vector store to search.
        query: User's question string.
        k: Number of results to return.

    Returns:
        List of (Document, score) tuples, sorted by relevance.
    """
    return vector_store.similarity_search_with_score(query, k=k)
