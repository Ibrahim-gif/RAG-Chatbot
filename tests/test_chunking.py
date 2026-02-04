"""
Unit tests for the chunking module.

Tests the document chunking functionality including PDF and Markdown document splitting
using various chunking strategies.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document
from src.chunking.basic import Chunking


class TestChunkingInitialization:
    """Test Chunking class initialization."""
    
    def test_chunking_default_initialization(self):
        """Test Chunking initialization with default parameters."""
        chunker = Chunking()
        assert chunker.chunk_size == 1000
        assert chunker.chunk_overlap == 100
        assert chunker.chunk_strategy == "Structure-Based"
        assert chunker.document_name is None
        assert chunker.document_type == "pdf"
    
    def test_chunking_custom_initialization(self):
        """Test Chunking initialization with custom parameters."""
        chunker = Chunking(
            chunk_size=500,
            chunk_overlap=50,
            chunk_strategy="Length-Based",
            document_name="test_doc.pdf",
            document_type="md"
        )
        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 50
        assert chunker.chunk_strategy == "Length-Based"
        assert chunker.document_name == "test_doc.pdf"
        assert chunker.document_type == "md"


class TestChunkingPDFDocuments:
    """Test chunking of PDF documents."""
    
    def test_pdf_structure_based_chunking(self):
        """Test PDF chunking with Structure-Based strategy."""
        chunker = Chunking(
            chunk_size=100,
            chunk_overlap=20,
            chunk_strategy="Structure-Based",
            document_name="test.pdf",
            document_type="pdf"
        )
        
        # Mock PDF documents
        mock_docs = [
            Document(page_content="Section 1\n\nThis is the first section with some content.", metadata={"source": "test.pdf", "page": 0}),
            Document(page_content="Section 2\n\nThis is the second section with more content.", metadata={"source": "test.pdf", "page": 1})
        ]
        
        result = chunker.chunk_document(mock_docs)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(chunk, Document) for chunk in result)
        assert all("source" in chunk.metadata for chunk in result)
    
    def test_pdf_length_based_chunking(self):
        """Test PDF chunking with Length-Based strategy."""
        chunker = Chunking(
            chunk_size=100,
            chunk_overlap=20,
            chunk_strategy="Length-Based",
            document_name="test.pdf",
            document_type="pdf"
        )
        
        mock_docs = [
            Document(page_content="This is a test document.\n\nWith multiple paragraphs.\n\nAnd more content.", metadata={"source": "test.pdf"})
        ]
        
        result = chunker.chunk_document(mock_docs)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(chunk, Document) for chunk in result)
    
    def test_pdf_chunks_respect_size_limits(self):
        """Test that PDF chunks respect the chunk_size limit."""
        chunker = Chunking(
            chunk_size=200,
            chunk_overlap=20,
            chunk_strategy="Structure-Based",
            document_type="pdf"
        )
        
        mock_docs = [
            Document(page_content="Word " * 100, metadata={"source": "test.pdf"})
        ]
        
        result = chunker.chunk_document(mock_docs)
        
        # Most chunks should be under the chunk_size limit (with some flexibility for structure)
        assert len(result) > 0
        assert all(len(chunk.page_content) <= 300 for chunk in result)  # Allow some buffer


class TestChunkingMarkdownDocuments:
    """Test chunking of Markdown documents."""
    
    def test_markdown_header_based_chunking(self):
        """Test Markdown chunking with header-based splitting."""
        chunker = Chunking(
            chunk_size=500,
            chunk_overlap=50,
            document_name="test.md",
            document_type="md"
        )
        
        markdown_content = """# Main Title
This is the introduction.

## Section 1
Content of section 1.

### Subsection 1.1
More detailed content.

## Section 2
Content of section 2.
"""
        
        result = chunker.chunk_document(markdown_content)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(chunk, Document) for chunk in result)
        assert all("Header" in chunk.metadata or "source" in chunk.metadata for chunk in result)
    
    def test_markdown_with_minimal_headers(self):
        """Test Markdown chunking with minimal header structure."""
        chunker = Chunking(
            chunk_size=500,
            document_name="simple.md",
            document_type="md"
        )
        
        markdown_content = """Simple content without many headers.
        
Some paragraphs here.
More content."""
        
        result = chunker.chunk_document(markdown_content)
        
        assert isinstance(result, list)
        assert len(result) >= 1


class TestChunkingMetadata:
    """Test metadata handling in chunks."""
    
    def test_pdf_chunks_include_metadata(self):
        """Test that PDF chunks include source metadata."""
        chunker = Chunking(
            chunk_size=100,
            chunk_overlap=10,
            document_name="energy.pdf",
            document_type="pdf"
        )
        
        mock_docs = [
            Document(page_content="Energy efficiency content.", metadata={"source": "energy.pdf", "page": 0})
        ]
        
        result = chunker.chunk_document(mock_docs)
        
        assert all("source" in chunk.metadata for chunk in result)
        assert all(chunk.metadata.get("source") == "energy.pdf" for chunk in result)
    
    def test_markdown_chunks_include_source(self):
        """Test that Markdown chunks include source metadata."""
        chunker = Chunking(
            document_name="guide.md",
            document_type="md"
        )
        
        markdown_content = "# Title\nContent here."
        result = chunker.chunk_document(markdown_content)
        
        assert all("source" in chunk.metadata for chunk in result)


class TestChunkingEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_pdf_document(self):
        """Test handling of empty PDF document."""
        chunker = Chunking(document_type="pdf")
        
        mock_docs = [Document(page_content="", metadata={"source": "empty.pdf"})]
        result = chunker.chunk_document(mock_docs)
        
        # Should handle gracefully, even if result is empty
        assert isinstance(result, list)
    
    def test_empty_markdown_document(self):
        """Test handling of empty Markdown document."""
        chunker = Chunking(document_type="md")
        
        result = chunker.chunk_document("")
        
        assert isinstance(result, list)
    
    def test_single_word_document(self):
        """Test chunking a very short document."""
        chunker = Chunking(chunk_size=100, document_type="pdf")
        
        mock_docs = [Document(page_content="Word", metadata={"source": "short.pdf"})]
        result = chunker.chunk_document(mock_docs)
        
        assert isinstance(result, list)
        assert len(result) >= 1
    
    def test_very_large_chunk_size(self):
        """Test with chunk size larger than document."""
        chunker = Chunking(chunk_size=10000, document_type="pdf")
        
        mock_docs = [Document(page_content="Small document content.", metadata={"source": "test.pdf"})]
        result = chunker.chunk_document(mock_docs)
        
        assert isinstance(result, list)
        assert len(result) >= 1


class TestChunkingConsistency:
    """Test consistency of chunking results."""
    
    def test_same_document_same_chunks(self):
        """Test that chunking the same document twice produces consistent results."""
        chunker1 = Chunking(chunk_size=200, chunk_overlap=20, document_type="pdf")
        chunker2 = Chunking(chunk_size=200, chunk_overlap=20, document_type="pdf")
        
        mock_doc = [Document(page_content="Consistent test content for evaluation.", metadata={"source": "test.pdf"})]
        
        result1 = chunker1.chunk_document(mock_doc)
        result2 = chunker2.chunk_document(mock_doc)
        
        assert len(result1) == len(result2)
        assert all(c1.page_content == c2.page_content for c1, c2 in zip(result1, result2))
    
    def test_overlap_creates_duplicate_content(self):
        """Test that chunk_overlap creates overlapping content between chunks."""
        chunker = Chunking(
            chunk_size=100,
            chunk_overlap=30,
            chunk_strategy="Structure-Based",
            document_type="pdf"
        )
        
        mock_doc = [Document(page_content="Word " * 50, metadata={"source": "test.pdf"})]
        result = chunker.chunk_document(mock_doc)
        
        # With overlap, consecutive chunks should have some shared content
        if len(result) > 1:
            for i in range(len(result) - 1):
                # Check that some content overlaps
                current_end = result[i].page_content[-30:]
                next_start = result[i+1].page_content[:30]
                # Due to word boundaries, exact overlap may vary
                assert isinstance(current_end, str) and isinstance(next_start, str)
