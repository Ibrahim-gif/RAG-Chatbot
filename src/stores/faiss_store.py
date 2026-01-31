# src/rag/stores/faiss_store.py
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
    Wrapper around LangChain FAISS vector store.
    Stores Documents with page_content + metadata.
    """

    def __init__(self, embedding_fn=None):
        """
        embedding_fn: something that behaves like LangChain embeddings
        (e.g., OpenAIEmbeddings or our wrapper's internal client)
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
        self._ensure_ready()
        self._vs.add_texts(texts=texts, metadatas=metadatas)
        self._vs.save_local(self.index_dir)

    def similarity_search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        self._ensure_ready()
        #'similarity' (default),'mmr', 'similarity_score_threshold'
        retriever = self._vs.as_retriever(search_type="similarity", search_kwargs={"k": k})
        # retriever.invoke(query, filter={"page": 0})
        return retriever.invoke(query)

    def _ensure_ready(self) -> None:
        if self._vs is None:
            raise RuntimeError("FaissStore is not initialized. Call load_or_create() first.")
