"""BM25 keyword search index.

Provides the keyword-matching half of the hybrid search pipeline.
BM25 finds documents with exact term overlap, complementing
FAISS's semantic (meaning-based) search.
"""

import re
from rank_bm25 import BM25Okapi


class BM25Index:
    """Lightweight BM25 index over LangChain Document objects.

    Usage:
        index = BM25Index(documents)      # build from a list
        index.add_documents(new_docs)      # add incrementally
        results = index.search(query, k=5) # search
    """

    def __init__(self, documents=None):
        self.documents = []
        self.index = None
        if documents:
            self.add_documents(documents)

    def add_documents(self, documents):
        """Add documents and rebuild the BM25 index.

        Args:
            documents: List of LangChain Document objects.
        """
        self.documents.extend(documents)
        self._rebuild()

    def _rebuild(self):
        """Rebuild the internal BM25 index from all stored documents."""
        corpus = [self._tokenize(doc.page_content) for doc in self.documents]
        self.index = BM25Okapi(corpus)

    def search(self, query, k=5):
        """Search for the most relevant documents by keyword overlap.

        Args:
            query: The search query string.
            k: Number of results to return.

        Returns:
            List of (Document, bm25_score) tuples, highest score first.
        """
        if self.index is None or not self.documents:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.index.get_scores(tokenized_query)

        # Get top-k indices sorted by score descending
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:k]

        return [
            (self.documents[i], float(scores[i]))
            for i in top_indices
            if scores[i] > 0
        ]

    @staticmethod
    def _tokenize(text):
        """Simple word tokenizer: lowercase + extract alphanumeric tokens."""
        return re.findall(r"\w+", text.lower())
