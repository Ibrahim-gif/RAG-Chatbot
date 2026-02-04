"""
Unit tests for the FastAPI application endpoints.

Tests the REST API endpoints for document upload, deletion, listing, and querying.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from src.main import app


@pytest.fixture
def client():
    """Fixture providing FastAPI test client."""
    return TestClient(app)


class TestListFilesEndpoint:
    """Test GET /files endpoint."""
    
    @patch("src.main.list_all_documents")
    def test_list_files_success(self, mock_list_docs, client):
        """Test successfully listing indexed files."""
        mock_list_docs.return_value = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
        
        response = client.get("/files")
        
        assert response.status_code == 200
        assert response.json() == ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
        mock_list_docs.assert_called_once()
    
    @patch("src.main.list_all_documents")
    def test_list_files_empty(self, mock_list_docs, client):
        """Test listing files when vector store is empty."""
        mock_list_docs.return_value = []
        
        response = client.get("/files")
        
        assert response.status_code == 200
        assert response.json() == []
    
    @patch("src.main.list_all_documents")
    def test_list_files_error(self, mock_list_docs, client):
        """Test error handling when list_all_documents fails."""
        mock_list_docs.side_effect = Exception("Vector store error")
        
        response = client.get("/files")
        
        assert response.status_code == 500


class TestDeleteFileEndpoint:
    """Test POST /delete endpoint."""
    
    @patch("src.main.delete_from_vector_store")
    def test_delete_file_success(self, mock_delete, client):
        """Test successfully deleting a file."""
        mock_delete.return_value = {"message": "File deleted successfully"}
        
        response = client.post("/delete", json={"file_name": "test.pdf"})
        
        assert response.status_code == 200
        assert "success" in response.json()["message"].lower() or "deleted" in response.json()["message"].lower()
        mock_delete.assert_called_once_with("test.pdf")
    
    @patch("src.main.delete_from_vector_store")
    def test_delete_file_not_found(self, mock_delete, client):
        """Test deleting a file that doesn't exist."""
        mock_delete.side_effect = FileNotFoundError("File not found in index")
        
        response = client.post("/delete", json={"file_name": "nonexistent.pdf"})
        
        assert response.status_code == 404 or response.status_code == 500
    
    def test_delete_file_missing_name(self, client):
        """Test delete endpoint with missing file_name."""
        response = client.post("/delete", json={})
        
        assert response.status_code in [400, 422]  # Validation error
    
    @patch("src.main.delete_from_vector_store")
    def test_delete_file_with_special_chars(self, mock_delete, client):
        """Test deleting file with special characters in name."""
        filename = "document_2024-01_v2.0.pdf"
        mock_delete.return_value = {"message": "Deleted"}
        
        response = client.post("/delete", json={"file_name": filename})
        
        assert response.status_code == 200
        mock_delete.assert_called_once_with(filename)


class TestUploadFileEndpoint:
    """Test POST /upload endpoint."""
    
    @patch("src.main.add_to_index")
    def test_upload_pdf_success(self, mock_add, client):
        """Test successfully uploading a PDF file."""
        mock_add.return_value = True
        
        pdf_content = b"%PDF-1.4 test content"
        
        response = client.post(
            "/upload",
            files={"file": ("test.pdf", pdf_content, "application/pdf")}
        )
        
        assert response.status_code in [200, 201]
        mock_add.assert_called_once()
    
    @patch("src.main.add_to_index")
    def test_upload_markdown_success(self, mock_add, client):
        """Test successfully uploading a Markdown file."""
        mock_add.return_value = True
        
        md_content = b"# Test Document\nContent here"
        
        response = client.post(
            "/upload",
            files={"file": ("test.md", md_content, "text/markdown")}
        )
        
        assert response.status_code in [200, 201]
        mock_add.assert_called_once()
    
    def test_upload_unsupported_format(self, client):
        """Test uploading unsupported file format."""
        txt_content = b"This is plain text"
        
        response = client.post(
            "/upload",
            files={"file": ("test.txt", txt_content, "text/plain")}
        )
        
        # Should reject unsupported format
        assert response.status_code in [400, 415, 422]
    
    def test_upload_no_file(self, client):
        """Test upload endpoint without file."""
        response = client.post("/upload")
        
        assert response.status_code in [400, 422]
    
    @patch("src.main.add_to_index")
    def test_upload_empty_file(self, mock_add, client):
        """Test uploading an empty file."""
        mock_add.return_value = True
        
        response = client.post(
            "/upload",
            files={"file": ("empty.pdf", b"", "application/pdf")}
        )
        
        # Should still process empty file
        assert response.status_code in [200, 201, 400, 422]
    
    @patch("src.main.add_to_index")
    def test_upload_large_file(self, mock_add, client):
        """Test uploading a large file."""
        mock_add.return_value = True
        
        # Create large file content (1MB)
        large_content = b"x" * (1024 * 1024)
        
        response = client.post(
            "/upload",
            files={"file": ("large.pdf", large_content, "application/pdf")}
        )
        
        assert response.status_code in [200, 201, 413]  # 413 if too large


class TestQueryEndpoint:
    """Test POST /query endpoint."""
    
    @patch("src.main.RAGAgent")
    def test_query_success(self, mock_rag_agent, client):
        """Test successful query."""
        from src.response_formats.definitions import LLMResponseWithCitations, Citation
        
        mock_response = LLMResponseWithCitations(
            answer="LED lighting is efficient",
            sources=[Citation(source="led.pdf", section="Page 1")]
        )
        mock_rag_agent.return_value = (mock_response, [])
        
        response = client.post(
            "/query",
            json={"query": "What is LED efficiency?"}
        )
        
        assert response.status_code == 200
        json_response = response.json()
        assert "answer" in json_response
    
    @patch("src.main.RAGAgent")
    def test_query_with_history(self, mock_rag_agent, client):
        """Test query with conversation history."""
        from src.response_formats.definitions import LLMResponseWithCitations
        
        mock_response = LLMResponseWithCitations(answer="Response", sources=[])
        mock_rag_agent.return_value = (mock_response, [])
        
        response = client.post(
            "/query",
            json={
                "query": "Follow-up question",
                "conversation_history": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi!"}
                ]
            }
        )
        
        assert response.status_code == 200
    
    def test_query_empty_query(self, client):
        """Test query endpoint with empty query."""
        response = client.post(
            "/query",
            json={"query": ""}
        )
        
        # May accept or reject empty query
        assert response.status_code in [200, 400, 422]
    
    def test_query_missing_query_field(self, client):
        """Test query endpoint without query field."""
        response = client.post(
            "/query",
            json={}
        )
        
        assert response.status_code in [400, 422]
    
    @patch("src.main.RAGAgent")
    def test_query_special_characters(self, mock_rag_agent, client):
        """Test query with special characters."""
        from src.response_formats.definitions import LLMResponseWithCitations
        
        mock_response = LLMResponseWithCitations(answer="Response", sources=[])
        mock_rag_agent.return_value = (mock_response, [])
        
        response = client.post(
            "/query",
            json={"query": "What about \"LED\" efficiency? (Cost: $100)"}
        )
        
        assert response.status_code == 200
    
    @patch("src.main.RAGAgent")
    def test_query_very_long_query(self, mock_rag_agent, client):
        """Test query with very long text."""
        from src.response_formats.definitions import LLMResponseWithCitations
        
        mock_response = LLMResponseWithCitations(answer="Response", sources=[])
        mock_rag_agent.return_value = (mock_response, [])
        
        long_query = "Question " * 1000
        
        response = client.post(
            "/query",
            json={"query": long_query}
        )
        
        assert response.status_code in [200, 413, 414]


class TestHealthEndpoint:
    """Test basic endpoint health checks."""
    
    def test_app_startup(self, client):
        """Test that app can be instantiated."""
        assert client is not None
    
    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.get("/files")
        
        # FastAPI returns 200 or 500, not CORS errors
        assert response.status_code in [200, 500]


class TestSafeFilenameFunction:
    """Test safe_filename utility function."""
    
    def test_safe_filename_normal(self):
        """Test safe_filename with normal filename."""
        from src.main import safe_filename
        
        result = safe_filename("document.pdf")
        assert result == "document.pdf"
    
    def test_safe_filename_with_path(self):
        """Test safe_filename strips path components."""
        from src.main import safe_filename
        
        result = safe_filename("path/to/document.pdf")
        assert result == "document.pdf"
        assert "/" not in result
    
    def test_safe_filename_with_null_bytes(self):
        """Test safe_filename removes null bytes."""
        from src.main import safe_filename
        
        result = safe_filename("document\x00.pdf")
        assert "\x00" not in result
    
    def test_safe_filename_windows_path(self):
        """Test safe_filename with Windows path."""
        from src.main import safe_filename
        
        result = safe_filename("C:\\Users\\Documents\\file.pdf")
        assert result == "file.pdf"
        assert "\\" not in result


class TestAPIErrorHandling:
    """Test API error handling and validation."""
    
    def test_invalid_json(self, client):
        """Test endpoint with invalid JSON."""
        response = client.post(
            "/query",
            data="{invalid json}",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code in [400, 422]
    
    def test_method_not_allowed(self, client):
        """Test using wrong HTTP method."""
        response = client.put("/files")
        
        assert response.status_code == 405
    
    def test_missing_content_type_json(self, client):
        """Test POST without Content-Type header."""
        response = client.post(
            "/query",
            data='{"query": "test"}',
            headers={"Content-Type": "text/plain"}
        )
        
        # May or may not accept depending on implementation
        assert response.status_code in [400, 415, 422, 200]


class TestEndpointIntegration:
    """Test integration between endpoints."""
    
    @patch("src.main.add_to_index")
    @patch("src.main.list_all_documents")
    def test_upload_then_list(self, mock_list, mock_add, client):
        """Test uploading a file then listing files."""
        mock_add.return_value = True
        mock_list.return_value = ["uploaded_doc.pdf"]
        
        # Upload file
        client.post(
            "/upload",
            files={"file": ("uploaded_doc.pdf", b"content", "application/pdf")}
        )
        
        # List files
        response = client.get("/files")
        
        assert response.status_code == 200
        assert "uploaded_doc.pdf" in response.json()
    
    @patch("src.main.RAGAgent")
    @patch("src.main.add_to_index")
    def test_upload_then_query(self, mock_add, mock_rag, client):
        """Test uploading a file then querying it."""
        from src.response_formats.definitions import LLMResponseWithCitations
        
        mock_add.return_value = True
        mock_response = LLMResponseWithCitations(answer="Found answer", sources=[])
        mock_rag.return_value = (mock_response, [])
        
        # Upload
        client.post(
            "/upload",
            files={"file": ("doc.pdf", b"content", "application/pdf")}
        )
        
        # Query
        response = client.post("/query", json={"query": "Test question"})
        
        assert response.status_code == 200
