# src/rag/chunking/basic.py
from __future__ import annotations
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.documents import Document

class Chunking:
    def __init__(self, chunk_size: int =1000, chunk_overlap: int = 200, chunk_strategy: str = "Structure-Based", document_name: str = None, document_type: str = "pdf"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_strategy = chunk_strategy
        self.document_name = document_name
        self.document_type = document_type
        
    def chunk_document(self, documents: Document | str) -> List[Document]:
        
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
        
