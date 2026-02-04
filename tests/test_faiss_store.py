"""
Unit tests for the FAISS vector store module.

Tests the FAISS vector store wrapper for document embeddings and retrieval.
"""

import pytest
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
from src.stores.faiss_store import FaissStore
from langchain_core.documents import Document


class TestFaissStoreInitialization:
    """Test FaissStore initialization."""
    
    @patch("src.stores.faiss_store.FAISS")
    @patch("src.stores.faiss_store.faiss")
    def test_faiss_store_new_index(self, mock_faiss, mock_langchain_faiss):
        """Test FaissStore initialization with new index."""
        # Mock the embedding function
        mock_embedding_fn = MagicMock()
        mock_embedding_fn.embed_query.return_value = [0.1] * 1536
        
        # Mock faiss.IndexFlatL2
        mock_index = MagicMock()
        mock_faiss.IndexFlatL2.return_value = mock_index
        
        # Mock LangChain FAISS initialization
        mock_langchain_faiss.return_value = MagicMock()
        
        with patch("src.stores.faiss_store.os.path.exists", return_value=False):
            store = FaissStore(embedding_fn=mock_embedding_fn)
            
            assert store.embedding_fn == mock_embedding_fn
            assert store.index_dir == Path("faiss_index").resolve()
            mock_faiss.IndexFlatL2.assert_called_once()
    
    @patch("src.stores.faiss_store.FAISS")
    def test_faiss_store_load_existing_index(self, mock_langchain_faiss):
        """Test FaissStore initialization with existing index."""
        mock_embedding_fn = MagicMock()
        mock_vs = MagicMock()
        mock_langchain_faiss.load_local.return_value = mock_vs
        
        with patch("src.stores.faiss_store.os.path.exists", return_value=True):
            store = FaissStore(embedding_fn=mock_embedding_fn)
            
            assert store._vs == mock_vs
            mock_langchain_faiss.load_local.assert_called_once()
    
    def test_faiss_store_missing_embedding_fn(self):
        """Test FaissStore initialization without embedding function."""
        with pytest.raises(AttributeError):
            FaissStore(embedding_fn=None)


class TestFaissStoreAddDocuments:
    """Test adding documents to FAISS store."""
    
    @patch("src.stores.faiss_store.FAISS")
    @patch("src.stores.faiss_store.faiss")
    def test_add_single_document(self, mock_faiss, mock_langchain_faiss):
        """Test adding a single document."""
        mock_embedding_fn = MagicMock()
        mock_embedding_fn.embed_query.return_value = [0.1] * 1536
        
        mock_index = MagicMock()
        mock_faiss.IndexFlatL2.return_value = mock_index
        mock_vs = MagicMock()
        mock_langchain_faiss.return_value = mock_vs
        
        with patch("src.stores.faiss_store.os.path.exists", return_value=False):
            store = FaissStore(embedding_fn=mock_embedding_fn)
            store._vs = mock_vs
            
            texts = ["This is a test document."]
            metadatas = [{"source": "test.pdf", "page": 0}]
            
            store.add_documents(texts, metadatas)
            
            mock_vs.add_texts.assert_called_once_with(texts=texts, metadatas=metadatas)
            mock_vs.save_local.assert_called_once()
    
    @patch("src.stores.faiss_store.FAISS")
    @patch("src.stores.faiss_store.faiss")
    def test_add_multiple_documents(self, mock_faiss, mock_langchain_faiss):
        """Test adding multiple documents."""
        mock_embedding_fn = MagicMock()
        mock_embedding_fn.embed_query.return_value = [0.1] * 1536
        
        mock_vs = MagicMock()
        mock_langchain_faiss.return_value = mock_vs
        
        with patch("src.stores.faiss_store.os.path.exists", return_value=False):
            with patch("src.stores.faiss_store.faiss.IndexFlatL2"):
                store = FaissStore(embedding_fn=mock_embedding_fn)
                store._vs = mock_vs
                
                texts = ["Document 1", "Document 2", "Document 3"]
                metadatas = [
                    {"source": "file1.pdf", "page": 0},
                    {"source": "file2.pdf", "page": 0},
                    {"source": "file3.pdf", "page": 1}
                ]
                
                store.add_documents(texts, metadatas)
                
                assert mock_vs.add_texts.call_count == 1
                assert mock_vs.save_local.call_count == 1
    
    @patch("src.stores.faiss_store.FAISS")
    @patch("src.stores.faiss_store.faiss")
    def test_add_documents_with_empty_list(self, mock_faiss, mock_langchain_faiss):
        """Test adding empty list of documents."""
        mock_embedding_fn = MagicMock()
        mock_embedding_fn.embed_query.return_value = [0.1] * 1536
        
        mock_vs = MagicMock()
        mock_langchain_faiss.return_value = mock_vs
        
        with patch("src.stores.faiss_store.os.path.exists", return_value=False):
            with patch("src.stores.faiss_store.faiss.IndexFlatL2"):
                store = FaissStore(embedding_fn=mock_embedding_fn)
                store._vs = mock_vs
                
                store.add_documents([], [])
                
                mock_vs.add_texts.assert_called_once_with(texts=[], metadatas=[])


class TestFaissStoreSimilaritySearch:
    """Test similarity search functionality."""
    
    pass


class TestFaissStoreDocumentDeletion:
    """Test document deletion functionality."""
    
    @patch("src.stores.faiss_store.FAISS")
    @patch("src.stores.faiss_store.faiss")
    def test_delete_single_document(self, mock_faiss, mock_langchain_faiss):
        """Test deleting a single document."""
        mock_embedding_fn = MagicMock()
        mock_embedding_fn.embed_query.return_value = [0.1] * 1536
        
        # Create mock documents with the proper structure
        mock_doc = Document(
            page_content="Test content",
            metadata={"source": "data/docs/test.pdf"},
            id="test-doc-1"
        )
        
        mock_vs = MagicMock()
        mock_vs.docstore._dict.values.return_value = [mock_doc]
        mock_langchain_faiss.return_value = mock_vs
        
        with patch("src.stores.faiss_store.os.path.exists", return_value=False):
            with patch("src.stores.faiss_store.faiss.IndexFlatL2"):
                store = FaissStore(embedding_fn=mock_embedding_fn)
                store._vs = mock_vs
                
                store.delete("test.pdf")
                
                mock_vs.delete.assert_called_once_with(ids=["test-doc-1"])
                mock_vs.save_local.assert_called_once()
    
    @patch("src.stores.faiss_store.FAISS")
    @patch("src.stores.faiss_store.faiss")
    def test_delete_multiple_documents(self, mock_faiss, mock_langchain_faiss):
        """Test deleting multiple documents."""
        mock_embedding_fn = MagicMock()
        mock_embedding_fn.embed_query.return_value = [0.1] * 1536
        
        mock_vs = MagicMock()
        mock_langchain_faiss.return_value = mock_vs
        
        with patch("src.stores.faiss_store.os.path.exists", return_value=False):
            with patch("src.stores.faiss_store.faiss.IndexFlatL2"):
                store = FaissStore(embedding_fn=mock_embedding_fn)
                store._vs = mock_vs
                
                store.delete("doc1.pdf")
                store.delete("doc2.pdf")
                
                assert mock_vs.delete.call_count == 2


class TestFaissStoreIndexPersistence:
    """Test index persistence functionality."""
    
    @patch("src.stores.faiss_store.FAISS")
    @patch("src.stores.faiss_store.faiss")
    def test_save_index_after_add(self, mock_faiss, mock_langchain_faiss):
        """Test that index is saved after adding documents."""
        mock_embedding_fn = MagicMock()
        mock_embedding_fn.embed_query.return_value = [0.1] * 1536
        
        mock_vs = MagicMock()
        mock_langchain_faiss.return_value = mock_vs
        
        with patch("src.stores.faiss_store.os.path.exists", return_value=False):
            with patch("src.stores.faiss_store.faiss.IndexFlatL2"):
                store = FaissStore(embedding_fn=mock_embedding_fn)
                store._vs = mock_vs
                
                store.add_documents(["text"], [{"source": "test.pdf"}])
                
                # Verify save_local was called
                mock_vs.save_local.assert_called()
    
    @patch("src.stores.faiss_store.FAISS")
    @patch("src.stores.faiss_store.faiss")
    def test_load_persisted_index(self, mock_faiss, mock_langchain_faiss):
        """Test loading a persisted index."""
        mock_embedding_fn = MagicMock()
        mock_embedding_fn.embed_query.return_value = [0.1] * 1536
        
        mock_vs = MagicMock()
        mock_langchain_faiss.load_local.return_value = mock_vs
        
        with patch("src.stores.faiss_store.os.path.exists", return_value=True):
            store = FaissStore(embedding_fn=mock_embedding_fn)
            
            assert store._vs == mock_vs
            mock_langchain_faiss.load_local.assert_called_once()


class TestFaissStoreEdgeCases:
    """Test edge cases and error handling."""
    
    @patch("src.stores.faiss_store.FAISS")
    @patch("src.stores.faiss_store.faiss")
    def test_search_empty_store(self, mock_faiss, mock_langchain_faiss):
        """Test searching an empty store."""
        mock_embedding_fn = MagicMock()
        mock_embedding_fn.embed_query.return_value = [0.1] * 1536
        
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = []
        # Also mock as_retriever().invoke() for consistency
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        mock_vs.as_retriever.return_value = mock_retriever
        mock_langchain_faiss.return_value = mock_vs
        
        with patch("src.stores.faiss_store.os.path.exists", return_value=False):
            with patch("src.stores.faiss_store.faiss.IndexFlatL2"):
                store = FaissStore(embedding_fn=mock_embedding_fn)
                store._vs = mock_vs
                
                results = store.similarity_search("query")
                
                assert results == []
    
    @patch("src.stores.faiss_store.FAISS")
    @patch("src.stores.faiss_store.faiss")
    def test_add_documents_mismatched_lengths(self, mock_faiss, mock_langchain_faiss):
        """Test adding documents with mismatched text/metadata lengths."""
        mock_embedding_fn = MagicMock()
        mock_embedding_fn.embed_query.return_value = [0.1] * 1536
        
        mock_vs = MagicMock()
        mock_langchain_faiss.return_value = mock_vs
        
        with patch("src.stores.faiss_store.os.path.exists", return_value=False):
            with patch("src.stores.faiss_store.faiss.IndexFlatL2"):
                store = FaissStore(embedding_fn=mock_embedding_fn)
                store._vs = mock_vs
                
                # This might be handled by langchain, but test graceful behavior
                texts = ["Text 1", "Text 2"]
                metadatas = [{"source": "test.pdf"}]  # Only one metadata
                
                store.add_documents(texts, metadatas)
                
                # Should still attempt to add
                mock_vs.add_texts.assert_called_once()
    
    @patch("src.stores.faiss_store.FAISS")
    @patch("src.stores.faiss_store.faiss")
    def test_search_with_very_long_query(self, mock_faiss, mock_langchain_faiss):
        """Test search with very long query string."""
        mock_embedding_fn = MagicMock()
        mock_embedding_fn.embed_query.return_value = [0.1] * 1536
        
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = []
        # Also mock as_retriever().invoke() for consistency
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        mock_vs.as_retriever.return_value = mock_retriever
        mock_langchain_faiss.return_value = mock_vs
        
        with patch("src.stores.faiss_store.os.path.exists", return_value=False):
            with patch("src.stores.faiss_store.faiss.IndexFlatL2"):
                store = FaissStore(embedding_fn=mock_embedding_fn)
                store._vs = mock_vs
                
                long_query = "word " * 10000
                results = store.similarity_search(long_query)
                
                assert isinstance(results, list)
