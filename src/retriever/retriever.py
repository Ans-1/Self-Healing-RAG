"""
Retriever — wraps ChromaDB with a similarity search interface.

Phase 2 will expand this with score thresholding and reranking.
"""
from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma
from src.ingestion.embedder import get_vector_store
from src.config import load_config


class Retriever:
    def __init__(self, vector_store: Chroma | None = None):
        self._vector_store = vector_store or get_vector_store()
        config = load_config()
        self._top_k = config["retrieval"]["top_k"]

    def retrieve(self, query: str, top_k: int | None = None) -> List[Document]:
        """Return top-k relevant documents for a query."""
        k = top_k or self._top_k
        return self._vector_store.similarity_search(query, k=k)

    def retrieve_with_scores(
        self, query: str, top_k: int | None = None
    ) -> List[tuple[Document, float]]:
        """Return (document, relevance_score) pairs. Lower score = more similar in L2."""
        k = top_k or self._top_k
        return self._vector_store.similarity_search_with_score(query, k=k)
