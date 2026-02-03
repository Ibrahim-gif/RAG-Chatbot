# src/rag/stores/faiss_store.py
"""
FAISS Vector Store wrapper module.

This module provides a wrapper around LangChain's FAISS vector store for
persistent local storage and retrieval of document embeddings. Supports
operations like adding documents, similarity search, and deletion.

"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple
from pathlib import Path

import faiss
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
import logging

class FaissStore:
    """
    FAISS vector store wrapper for document embeddings and retrieval.
    
    Manages local FAISS indices for storing and retrieving document embeddings.
    Supports adding new documents, similarity searches, and document deletion.
    Automatically persists changes to disk.
    
    Attributes:
        index_dir (Path): Directory where the FAISS index is stored.
        embedding_fn: Embedding function to use for vectorization.
        _vs (Optional[FAISS]): Internal LangChain FAISS vector store instance.
    """

    def __init__(self, embedding_fn=None):
        """
        Initialize or load a FAISS vector store.
        
        Creates a new FAISS index if one doesn't exist, or loads an existing
        index from disk if available.
        
        Args:
            embedding_fn: An embedding function that has an embed_query() method
                         (e.g., OpenAIEmbeddings or OpenAIEmbedder client).
                         
        Raises:
            ValueError: If embedding_fn is None.
        """
        self.index_dir = Path("faiss_index").resolve()
        self.embedding_fn = embedding_fn
        self._vs: Optional[FAISS] = None
        
        if os.path.exists(self.index_dir):
            logging.info(f"Loading existing FAISS index from {self.index_dir}")
            self._vs = FAISS.load_local(
                folder_path=self.index_dir,
                embeddings=self.embedding_fn,
                allow_dangerous_deserialization=True,
            )
        else: 
            index = faiss.IndexFlatL2(len(self.embedding_fn.embed_query("hello world")))
            logging.info(f"Creating new FAISS index at {self.index_dir}")

            self._vs = FAISS(
                embedding_function=self.embedding_fn,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )

    def add_documents(self, texts: List[str], metadatas: List[dict]) -> None:
        """
        Add documents (chunks) to the FAISS index.
        
        Args:
            texts (List[str]): List of document texts/chunks to add.
            metadatas (List[dict]): List of metadata dictionaries corresponding to each text.
                                   Must include 'source' field identifying the document.
        
        Returns:
            None
        
        Raises:
            RuntimeError: If the vector store is not initialized.
        """
        self._ensure_ready()
        self._vs.add_texts(texts=texts, metadatas=metadatas)
        self._vs.save_local(self.index_dir)

    def similarity_search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        Perform a similarity search in the vector store.
        
        Finds the k most similar documents to the given query using cosine similarity
        in the embedding space.
        
        Args:
            query (str): The query text to search for.
            k (int, optional): Number of top results to return. Defaults to 5.
        
        Returns:
            List[Tuple[Document, float]]: A list of (Document, similarity_score) tuples.
                                          Documents are ranked by similarity (highest first).
        
        Raises:
            RuntimeError: If the vector store is not initialized.
        """
        self._ensure_ready()
        #'similarity' (default),'mmr', 'similarity_score_threshold'
        retriever = self._vs.as_retriever(search_type="similarity", search_kwargs={"k": k})
        # retriever.invoke(query, filter={"page": 0})
        return retriever.invoke(query)

    def _ensure_ready(self) -> None:
        """
        Ensure the vector store is initialized.
        
        Raises:
            RuntimeError: If the vector store is not initialized.
        """
        if self._vs is None:
            raise RuntimeError("FaissStore is not initialized. Call load_or_create() first.")
        
    def delete(self, file_name: str):
        """
        Delete all chunks from a specific document from the vector store.
        
        Removes all entries whose source metadata ends with the given filename
        and saves the updated index.
        
        Args:
            file_name (str): The name of the document to delete (e.g., 'policy.pdf').
        
        Returns:
            None
        
        Raises:
            RuntimeError: If the vector store is not initialized.
        """
        delete_list_ids = []
        for document in self._vs.docstore._dict.values():
            if str(document.metadata.get("source")).endswith(file_name):
                delete_list_ids.append(document.id)
        self._vs.delete(delete_list_ids)
        self._vs.save_local(self.index_dir)
