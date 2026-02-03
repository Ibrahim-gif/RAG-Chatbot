# src/rag/chunking/basic.py
"""
Document chunking module for splitting documents into manageable pieces.

This module provides the Chunking class which handles splitting PDF and Markdown
documents into chunks suitable for embedding and retrieval. It supports multiple
chunking strategies including structure-based, length-based, and markdown header-based splitting.

"""

from __future__ import annotations
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.documents import Document

class Chunking:
    """
    Document chunking class for splitting documents into manageable chunks.
    
    Supports multiple chunking strategies:
    - Structure-Based: RecursiveCharacterTextSplitter that preserves larger text units
    - Length-Based: CharacterTextSplitter with fixed chunk sizes
    - Markdown Header-Based: MarkdownHeaderTextSplitter for structured markdown documents
    
    Attributes:
        chunk_size (int): Maximum size of each chunk in characters.
        chunk_overlap (int): Number of overlapping characters between consecutive chunks.
        chunk_strategy (str): Strategy to use for chunking ('Structure-Based', 'Length-Based').
        document_name (str): Name/path of the document being chunked.
        document_type (str): Type of document ('pdf' or 'md').
    """
    
    def __init__(self, chunk_size: int =1000, chunk_overlap: int = 200, chunk_strategy: str = "Structure-Based", document_name: str = None, document_type: str = "pdf"):
        """
        Initialize the Chunking class.
        
        Args:
            chunk_size (int, optional): Maximum size of each chunk in characters. Defaults to 1000.
            chunk_overlap (int, optional): Number of overlapping characters between chunks. Defaults to 200.
            chunk_strategy (str, optional): Chunking strategy to use. Defaults to "Structure-Based".
                                           Options: "Structure-Based", "Length-Based"
            document_name (str, optional): Name or path of the document. Defaults to None.
            document_type (str, optional): Type of document ('pdf' or 'md'). Defaults to "pdf".
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_strategy = chunk_strategy
        self.document_name = document_name
        self.document_type = document_type
        
    def chunk_document(self, documents: Document | str) -> List[Document]:
        """
        Split a document into chunks based on the configured strategy.
        
        For PDF documents:
        - Structure-Based: Uses RecursiveCharacterTextSplitter to preserve document structure
        - Length-Based: Uses CharacterTextSplitter with fixed-size chunks
        
        For Markdown documents:
        - Uses MarkdownHeaderTextSplitter to split on headers (#, ##, ###)
        
        Args:
            documents (Document | str): The document(s) to chunk. Can be a LangChain Document object
                                        (for PDF) or a string (for Markdown).
        
        Returns:
            List[Document]: A list of LangChain Document objects representing the chunks.
                           Each chunk includes metadata with the source document name.
        
        Raises:
            ValueError: If an unsupported chunking strategy is specified for the document type.
        """
        
        splitter = None
        
        if self.document_type == "pdf": 
            """The RecursiveCharacterTextSplitter preserves larger text units when possible, 
                breaking them into smaller units only if they exceed the chunk size, 
                recursively down to the word level if needed."""
                
            if self.chunk_strategy == "Structure-Based":
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                )
            
            elif self.chunk_strategy == "Length-Based":
                splitter = CharacterTextSplitter(
                    separator="\n\n",
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    length_function=len,
                    is_separator_regex=False,
                )
            return splitter.split_documents(documents)
        
        elif self.document_type == "md":
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
            splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False)
            docs = splitter.split_text(documents)
            
            for doc in docs:
                doc.metadata["source"] = str(self.document_name)
            return docs 
        
