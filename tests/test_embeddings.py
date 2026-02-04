"""
Unit tests for the OpenAI embeddings module.

Tests the embedding functionality for documents and queries using OpenAI's embedding models.
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from src.embedding.openai_embeds import OpenAIEmbedder


class TestOpenAIEmbedderInitialization:
    """Test OpenAIEmbedder initialization."""
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key-123"})
    def test_embedder_default_initialization(self):
        """Test OpenAIEmbedder initialization with default parameters."""
        embedder = OpenAIEmbedder()
        assert embedder.model_name == "text-embedding-3-large"
        assert embedder.api_key == "test-key-123"
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key-123"})
    def test_embedder_custom_model(self):
        """Test OpenAIEmbedder initialization with custom model."""
        embedder = OpenAIEmbedder(model_name="text-embedding-3-small")
        assert embedder.model_name == "text-embedding-3-small"
        assert embedder.api_key == "test-key-123"
    
    def test_embedder_provided_api_key(self):
        """Test OpenAIEmbedder initialization with provided API key."""
        embedder = OpenAIEmbedder(api_key="provided-key-456")
        assert embedder.api_key == "provided-key-456"
    
    @patch.dict(os.environ, {}, clear=True)
    def test_embedder_missing_api_key(self):
        """Test OpenAIEmbedder raises error when API key is missing."""
        with pytest.raises(ValueError, match="OPENAI_API_KEY is not set"):
            OpenAIEmbedder()
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_embedder_provided_key_overrides_env(self):
        """Test that provided API key overrides environment variable."""
        embedder = OpenAIEmbedder(api_key="override-key")
        assert embedder.api_key == "override-key"


class TestOpenAIEmbedderDocuments:
    """Test document embedding functionality."""
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("src.embedding.openai_embeds.OpenAIEmbeddings")
    def test_embed_documents_single(self, mock_embeddings):
        """Test embedding a single document."""
        # Mock the OpenAIEmbeddings client
        mock_client = MagicMock()
        mock_embeddings.return_value = mock_client
        mock_client.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        
        embedder = OpenAIEmbedder()
        texts = ["Hello world"]
        
        result = embedder.embed_documents(texts)
        
        assert result == [[0.1, 0.2, 0.3]]
        mock_client.embed_documents.assert_called_once_with(texts)
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("src.embedding.openai_embeds.OpenAIEmbeddings")
    def test_embed_documents_multiple(self, mock_embeddings):
        """Test embedding multiple documents."""
        mock_client = MagicMock()
        mock_embeddings.return_value = mock_client
        expected_embeddings = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ]
        mock_client.embed_documents.return_value = expected_embeddings
        
        embedder = OpenAIEmbedder()
        texts = ["First document", "Second document", "Third document"]
        
        result = embedder.embed_documents(texts)
        
        assert result == expected_embeddings
        assert len(result) == 3
        mock_client.embed_documents.assert_called_once_with(texts)
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("src.embedding.openai_embeds.OpenAIEmbeddings")
    def test_embed_documents_empty_list(self, mock_embeddings):
        """Test embedding an empty list of documents."""
        mock_client = MagicMock()
        mock_embeddings.return_value = mock_client
        mock_client.embed_documents.return_value = []
        
        embedder = OpenAIEmbedder()
        texts = []
        
        result = embedder.embed_documents(texts)
        
        assert result == []
        mock_client.embed_documents.assert_called_once_with(texts)
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("src.embedding.openai_embeds.OpenAIEmbeddings")
    def test_embed_documents_special_characters(self, mock_embeddings):
        """Test embedding documents with special characters."""
        mock_client = MagicMock()
        mock_embeddings.return_value = mock_client
        mock_client.embed_documents.return_value = [[0.1, 0.2]]
        
        embedder = OpenAIEmbedder()
        texts = ["Energy efficiency: 💡", "Cost savings @ $100", "Quote: \"Hello\""]
        
        result = embedder.embed_documents(texts)
        
        assert result == [[0.1, 0.2]]
        mock_client.embed_documents.assert_called_once_with(texts)
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("src.embedding.openai_embeds.OpenAIEmbeddings")
    def test_embed_documents_long_text(self, mock_embeddings):
        """Test embedding very long documents."""
        mock_client = MagicMock()
        mock_embeddings.return_value = mock_client
        mock_client.embed_documents.return_value = [[0.1] * 1536]  # Typical embedding dimension
        
        embedder = OpenAIEmbedder()
        # Create a very long text
        long_text = "word " * 5000
        texts = [long_text]
        
        result = embedder.embed_documents(texts)
        
        assert len(result) == 1
        assert len(result[0]) == 1536


class TestOpenAIEmbedderQuery:
    """Test query embedding functionality."""
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("src.embedding.openai_embeds.OpenAIEmbeddings")
    def test_embed_query_simple(self, mock_embeddings):
        """Test embedding a simple query."""
        mock_client = MagicMock()
        mock_embeddings.return_value = mock_client
        mock_client.embed_query.return_value = [0.1, 0.2, 0.3]
        
        embedder = OpenAIEmbedder()
        query = "What is energy efficiency?"
        
        result = embedder.embed_query(query)
        
        assert result == [0.1, 0.2, 0.3]
        mock_client.embed_query.assert_called_once_with(query)
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("src.embedding.openai_embeds.OpenAIEmbeddings")
    def test_embed_query_empty(self, mock_embeddings):
        """Test embedding an empty query."""
        mock_client = MagicMock()
        mock_embeddings.return_value = mock_client
        mock_client.embed_query.return_value = [0.0] * 1536
        
        embedder = OpenAIEmbedder()
        query = ""
        
        result = embedder.embed_query(query)
        
        assert isinstance(result, list)
        mock_client.embed_query.assert_called_once_with(query)
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("src.embedding.openai_embeds.OpenAIEmbeddings")
    def test_embed_query_special_characters(self, mock_embeddings):
        """Test embedding query with special characters."""
        mock_client = MagicMock()
        mock_embeddings.return_value = mock_client
        mock_client.embed_query.return_value = [0.1, 0.2]
        
        embedder = OpenAIEmbedder()
        query = "Energy efficiency: How to save money? 🔌"
        
        result = embedder.embed_query(query)
        
        assert result == [0.1, 0.2]
        mock_client.embed_query.assert_called_once_with(query)
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("src.embedding.openai_embeds.OpenAIEmbeddings")
    def test_embed_query_multiple_sentences(self, mock_embeddings):
        """Test embedding a multi-sentence query."""
        mock_client = MagicMock()
        mock_embeddings.return_value = mock_client
        mock_client.embed_query.return_value = [0.1, 0.2, 0.3]
        
        embedder = OpenAIEmbedder()
        query = "What are the benefits of LED lighting? How much can I save? Is it worth the investment?"
        
        result = embedder.embed_query(query)
        
        assert result == [0.1, 0.2, 0.3]


class TestEmbeddingDimensions:
    """Test embedding dimensions and consistency."""
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("src.embedding.openai_embeds.OpenAIEmbeddings")
    def test_consistent_embedding_dimensions(self, mock_embeddings):
        """Test that all embeddings have consistent dimensions."""
        mock_client = MagicMock()
        mock_embeddings.return_value = mock_client
        embedding_dim = 1536
        mock_client.embed_documents.return_value = [
            [0.1] * embedding_dim,
            [0.2] * embedding_dim,
            [0.3] * embedding_dim
        ]
        
        embedder = OpenAIEmbedder()
        texts = ["Text 1", "Text 2", "Text 3"]
        
        result = embedder.embed_documents(texts)
        
        assert all(len(emb) == embedding_dim for emb in result)
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("src.embedding.openai_embeds.OpenAIEmbeddings")
    def test_query_and_doc_same_dimensions(self, mock_embeddings):
        """Test that query and document embeddings have same dimensions."""
        mock_client = MagicMock()
        mock_embeddings.return_value = mock_client
        embedding_dim = 1536
        mock_client.embed_documents.return_value = [[0.1] * embedding_dim]
        mock_client.embed_query.return_value = [0.2] * embedding_dim
        
        embedder = OpenAIEmbedder()
        
        doc_embedding = embedder.embed_documents(["Document"])[0]
        query_embedding = embedder.embed_query("Query")
        
        assert len(doc_embedding) == len(query_embedding)


class TestEmbedderModels:
    """Test different embedding models."""
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("src.embedding.openai_embeds.OpenAIEmbeddings")
    def test_large_model(self, mock_embeddings):
        """Test with text-embedding-3-large model."""
        embedder = OpenAIEmbedder(model_name="text-embedding-3-large")
        assert embedder.model_name == "text-embedding-3-large"
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("src.embedding.openai_embeds.OpenAIEmbeddings")
    def test_small_model(self, mock_embeddings):
        """Test with text-embedding-3-small model."""
        embedder = OpenAIEmbedder(model_name="text-embedding-3-small")
        assert embedder.model_name == "text-embedding-3-small"
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("src.embedding.openai_embeds.OpenAIEmbeddings")
    def test_custom_model(self, mock_embeddings):
        """Test with custom model name."""
        embedder = OpenAIEmbedder(model_name="text-embedding-custom")
        assert embedder.model_name == "text-embedding-custom"
