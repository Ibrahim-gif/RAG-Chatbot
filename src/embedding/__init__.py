"""
Embedding module for the RAG system.

Provides interfaces and implementations for generating vector embeddings
of documents and queries using various embedding models.
"""

from .openai_embeds import OpenAIEmbedder

__all__ = ["OpenAIEmbedder"]
