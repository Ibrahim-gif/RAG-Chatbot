"""
Document chunking module for the RAG system.

Provides strategies for splitting documents into manageable chunks suitable
for embedding and retrieval.
"""

from .basic import Chunking

__all__ = ["Chunking"]
