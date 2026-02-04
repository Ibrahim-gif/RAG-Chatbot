"""
Unit tests for the RAG pipeline module.

Tests the main RAG orchestration including document ingestion, routing, and generation.
"""

import pytest
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from src.rag.pipeline import add_to_index, RAGAgent, RAGGeneration
from src.response_formats.definitions import LLMResponseWithCitations, RAGRouterResponse
from langchain_core.documents import Document


class TestAddToIndex:
    """Test document ingestion and indexing."""
    
    @patch("src.rag.pipeline.FaissStore")
    @patch("src.rag.pipeline.OpenAIEmbedder")
    @patch("src.rag.pipeline.Chunking")
    @patch("src.rag.pipeline.PyPDFLoader")
    def test_add_pdf_to_index(self, mock_pdf_loader, mock_chunking, mock_embedder, mock_store):
        """Test adding a PDF document to index."""
        # Setup mocks
        mock_docs = [Document(page_content="Test content", metadata={"page": 0})]
        mock_pdf_loader.return_value.load.return_value = mock_docs
        
        mock_chunks = [Document(page_content="Chunk 1", metadata={"source": "test.pdf"})]
        mock_chunking.return_value.chunk_document.return_value = mock_chunks
        
        mock_embedder_instance = MagicMock()
        mock_embedder.return_value = mock_embedder_instance
        
        mock_store_instance = MagicMock()
        mock_store.return_value = mock_store_instance
        
        configs = {
            "chunking_config": {"chunk_size": 1000, "chunk_overlap": 100},
            "embedder_model_config": {"model": "text-embedding-3-large"}
        }
        
        result = add_to_index("test.pdf", document_type="pdf", configs=configs)
        
        assert result is True
        mock_pdf_loader.assert_called_once_with(file_path="test.pdf")
        mock_store_instance.add_documents.assert_called_once()
    
    @patch("src.rag.pipeline.FaissStore")
    @patch("src.rag.pipeline.OpenAIEmbedder")
    @patch("src.rag.pipeline.Chunking")
    @patch("builtins.open", create=True)
    def test_add_markdown_to_index(self, mock_open, mock_chunking, mock_embedder, mock_store):
        """Test adding a Markdown document to index."""
        # Setup mocks
        mock_open.return_value.__enter__.return_value.read.return_value = "# Title\nContent"
        
        mock_chunks = [Document(page_content="Markdown chunk", metadata={"source": "test.md"})]
        mock_chunking.return_value.chunk_document.return_value = mock_chunks
        
        mock_embedder_instance = MagicMock()
        mock_embedder.return_value = mock_embedder_instance
        
        mock_store_instance = MagicMock()
        mock_store.return_value = mock_store_instance
        
        configs = {
            "chunking_config": {"chunk_size": 1000, "chunk_overlap": 100},
            "embedder_model_config": {"model": "text-embedding-3-large"}
        }
        
        result = add_to_index("test.md", document_type="md", configs=configs)
        
        assert result is True
        mock_store_instance.add_documents.assert_called_once()
    
    @patch("src.rag.pipeline.Chunking")
    @patch("src.rag.pipeline.PyPDFLoader")
    def test_add_to_index_nonexistent_file(self, mock_pdf_loader, mock_chunking):
        """Test adding a nonexistent file raises error."""
        mock_pdf_loader.return_value.load.side_effect = FileNotFoundError("File not found")
        
        configs = {
            "chunking_config": {"chunk_size": 1000, "chunk_overlap": 100},
            "embedder_model_config": {"model": "text-embedding-3-large"}
        }
        
        with pytest.raises(FileNotFoundError):
            add_to_index("nonexistent.pdf", document_type="pdf", configs=configs)


class TestRAGAgent:
    """Test RAG agent routing logic."""
    
    @patch("src.rag.pipeline.RAGGeneration")
    @patch("src.rag.pipeline.OpenAIChatLLM")
    def test_rag_agent_fetch_false_direct_response(self, mock_llm_class, mock_rag_gen):
        """Test RAG agent returns direct response when fetch_vector_store=False."""
        # Setup mocks
        mock_llm_instance = MagicMock()
        mock_llm_class.return_value = mock_llm_instance
        
        # Router says no fetch needed
        mock_router_response = RAGRouterResponse(
            fetch_vector_store=False,
            retrieval_queries=None
        )
        mock_llm_instance.structured_generate.return_value = mock_router_response
        
        # Direct response
        mock_llm_instance.generate.return_value = "Direct answer"
        
        configs = {
            "llm_model_config": {"model": "gpt-4", "max_tokens": 1000, "temperature": 0.0},
            "retriever_config": {"k": 4},
            "templates": {
                "RAG_ROUTER_SYSTEM_PROMPT": "Router prompt",
                "AI_ASSISTANT_SYSTEM_PROMPT": "Assistant prompt"
            }
        }
        
        response, docs = RAGAgent(
            user_query="What is energy?",
            conversation_history=[],
            configs=configs
        )
        
        assert response == "Direct answer"
        assert docs is None
        mock_rag_gen.assert_not_called()
    
    @patch("src.rag.pipeline.RAGGeneration")
    @patch("src.rag.pipeline.OpenAIChatLLM")
    def test_rag_agent_fetch_true_retrieval_response(self, mock_llm_class, mock_rag_gen):
        """Test RAG agent proceeds to retrieval when fetch_vector_store=True."""
        mock_llm_instance = MagicMock()
        mock_llm_class.return_value = mock_llm_instance
        
        # Router says fetch is needed
        mock_router_response = RAGRouterResponse(
            fetch_vector_store=True,
            retrieval_queries=["energy efficiency", "LED benefits"]
        )
        mock_llm_instance.structured_generate.return_value = mock_router_response
        
        # RAG generation response
        mock_rag_response = LLMResponseWithCitations(
            answer="LED lighting is efficient",
            sources=[]
        )
        mock_docs = [{"page_content": "LED info"}]
        mock_rag_gen.return_value = (mock_rag_response, mock_docs)
        
        configs = {
            "llm_model_config": {"model": "gpt-4", "max_tokens": 1000, "temperature": 0.0},
            "retriever_config": {"k": 4},
            "templates": {
                "RAG_ROUTER_SYSTEM_PROMPT": "Router prompt",
                "AI_ASSISTANT_SYSTEM_PROMPT": "Assistant prompt"
            }
        }
        
        response, docs = RAGAgent(
            user_query="Tell me about LED efficiency",
            conversation_history=[],
            configs=configs
        )
        
        assert response == mock_rag_response
        assert docs == mock_docs
        mock_rag_gen.assert_called_once()
    
    @patch("src.rag.pipeline.OpenAIChatLLM")
    def test_rag_agent_with_conversation_history(self, mock_llm_class):
        """Test RAG agent preserves conversation history."""
        mock_llm_instance = MagicMock()
        mock_llm_class.return_value = mock_llm_instance
        
        mock_router_response = RAGRouterResponse(
            fetch_vector_store=False,
            retrieval_queries=None
        )
        mock_llm_instance.structured_generate.return_value = mock_router_response
        mock_llm_instance.generate.return_value = "Response"
        
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"}
        ]
        
        configs = {
            "llm_model_config": {"model": "gpt-4", "max_tokens": 1000, "temperature": 0.0},
            "retriever_config": {"k": 4},
            "templates": {
                "RAG_ROUTER_SYSTEM_PROMPT": "Router prompt",
                "AI_ASSISTANT_SYSTEM_PROMPT": "Assistant prompt"
            }
        }
        
        RAGAgent(
            user_query="Test",
            conversation_history=history,
            configs=configs
        )
        
        # Verify history was passed to LLM
        call_args = mock_llm_instance.structured_generate.call_args
        assert call_args.kwargs.get("messages") == history


class TestRAGGeneration:
    """Test RAG generation with retrieval."""
    
    @patch("src.rag.pipeline.FaissStore")
    @patch("src.rag.pipeline.OpenAIEmbedder")
    @patch("src.rag.pipeline.OpenAIChatLLM")
    def test_rag_generation_multiple_queries(self, mock_llm_class, mock_embedder, mock_store):
        """Test RAG generation with multiple retrieval queries."""
        mock_embedder_instance = MagicMock()
        mock_embedder.return_value = mock_embedder_instance
        
        mock_store_instance = MagicMock()
        mock_store_instance.similarity_search.return_value = []
        mock_store.return_value = mock_store_instance
        
        mock_llm = MagicMock()
        
        configs = {
            "templates": {"AI_ASSISTANT_SYSTEM_PROMPT": "Generate"}
        }
        
        RAGGeneration(
            user_query="Question",
            retriever_query=["Query 1", "Query 2", "Query 3"],
            k=4,
            llm=mock_llm,
            configs=configs
        )
        
        # Verify multiple searches were performed
        assert mock_store_instance.similarity_search.call_count == 3
    
    @patch("src.rag.pipeline.FaissStore")
    @patch("src.rag.pipeline.OpenAIEmbedder")
    @patch("src.rag.pipeline.OpenAIChatLLM")
    def test_rag_generation_fallback_to_user_query(self, mock_llm_class, mock_embedder, mock_store):
        """Test RAG generation falls back to user_query if retriever_query is None."""
        mock_embedder_instance = MagicMock()
        mock_embedder.return_value = mock_embedder_instance
        
        mock_store_instance = MagicMock()
        mock_store_instance.similarity_search.return_value = []
        mock_store.return_value = mock_store_instance
        
        mock_llm = MagicMock()
        
        configs = {
            "templates": {"AI_ASSISTANT_SYSTEM_PROMPT": "Generate"}
        }
        
        RAGGeneration(
            user_query="What is LED?",
            retriever_query=None,
            k=4,
            llm=mock_llm,
            configs=configs
        )
        
        # Should use user_query for search
        call_args = mock_store_instance.similarity_search.call_args
        assert call_args is not None


class TestRAGPipelineIntegration:
    """Test integration of RAG pipeline components."""
    
    @patch("src.rag.pipeline.RAGGeneration")
    @patch("src.rag.pipeline.OpenAIChatLLM")
    def test_full_rag_flow(self, mock_llm_class, mock_rag_gen):
        """Test complete RAG flow from query to response."""
        # Setup router
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm
        
        router_response = RAGRouterResponse(
            fetch_vector_store=True,
            retrieval_queries=["test query"]
        )
        mock_llm.structured_generate.return_value = router_response
        
        # Setup generation
        final_response = LLMResponseWithCitations(
            answer="Test answer",
            sources=[]
        )
        mock_rag_gen.return_value = (final_response, [])
        
        configs = {
            "llm_model_config": {"model": "gpt-4", "max_tokens": 1000, "temperature": 0.0},
            "retriever_config": {"k": 4},
            "templates": {
                "RAG_ROUTER_SYSTEM_PROMPT": "Router",
                "AI_ASSISTANT_SYSTEM_PROMPT": "Assistant"
            }
        }
        
        response, docs = RAGAgent(
            user_query="Test question",
            conversation_history=[],
            configs=configs
        )
        
        assert response.answer == "Test answer"


class TestRAGPipelineEdgeCases:
    """Test edge cases in RAG pipeline."""
    
    @patch("src.rag.pipeline.FaissStore")
    @patch("src.rag.pipeline.OpenAIEmbedder")
    @patch("src.rag.pipeline.OpenAIChatLLM")
    def test_rag_generation_empty_retriever_query_list(self, mock_llm_class, mock_embedder, mock_store):
        """Test RAG generation with empty retriever query list."""
        mock_embedder_instance = MagicMock()
        mock_embedder.return_value = mock_embedder_instance
        
        mock_store_instance = MagicMock()
        mock_store.return_value = mock_store_instance
        
        mock_llm = MagicMock()
        
        configs = {
            "templates": {"AI_ASSISTANT_SYSTEM_PROMPT": "Generate"}
        }
        
        RAGGeneration(
            user_query="Test",
            retriever_query=[],
            k=4,
            llm=mock_llm,
            configs=configs
        )
        
        # Should not perform any searches for empty query list
        mock_store_instance.similarity_search.assert_not_called()
    
    @patch("src.rag.pipeline.OpenAIChatLLM")
    def test_rag_agent_with_special_characters_in_query(self, mock_llm_class):
        """Test RAG agent handles special characters in query."""
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm
        
        router_response = RAGRouterResponse(
            fetch_vector_store=False,
            retrieval_queries=None
        )
        mock_llm.structured_generate.return_value = router_response
        mock_llm.generate.return_value = "Response"
        
        configs = {
            "llm_model_config": {"model": "gpt-4", "max_tokens": 1000, "temperature": 0.0},
            "retriever_config": {"k": 4},
            "templates": {
                "RAG_ROUTER_SYSTEM_PROMPT": "Router",
                "AI_ASSISTANT_SYSTEM_PROMPT": "Assistant"
            }
        }
        
        response, docs = RAGAgent(
            user_query="What about \"LED\" efficiency? (Cost: $100)",
            conversation_history=[],
            configs=configs
        )
        
        assert response is not None
