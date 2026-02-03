"""
Vector store module for the RAG system.

Provides interfaces and implementations for storing and retrieving document
embeddings using various vector database backends.
"""

from .faiss_store import FaissStore

__all__ = ["FaissStore"]
