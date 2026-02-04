"""
Comprehensive test suite for the RAG system.

This package contains unit tests for all modules of the Retrieval-Augmented Generation (RAG) system:
- test_chunking: Document chunking strategies
- test_embeddings: OpenAI embedding generation
- test_faiss_store: Vector store operations
- test_openai_llm: Language model interactions
- test_response_formats: Pydantic response schemas
- test_rag_pipeline: RAG orchestration and routing
- test_main: FastAPI endpoints

Run all tests with: pytest
Run specific test file: pytest tests/test_chunking.py
Run with coverage: pytest --cov=src tests/
"""

__version__ = "1.0.0"
