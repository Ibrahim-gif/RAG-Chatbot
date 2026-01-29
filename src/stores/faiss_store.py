# src/rag/stores/faiss_store.py
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import faiss
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore

class FaissStore:
    """
    Wrapper around LangChain FAISS vector store.
    Stores Documents with page_content + metadata.
    """

    def __init__(self, index_dir: str, embedding_fn):
        """
        embedding_fn: something that behaves like LangChain embeddings
                      (e.g., OpenAIEmbeddings or our wrapper's internal client)
        """
        self.index_dir = index_dir
        self.embedding_fn = embedding_fn
        self._vs: Optional[FAISS] = None

    def load_or_create(self) -> None:
        if os.path.exists(self.index_dir) and os.listdir(self.index_dir):
            self._vs = FAISS.load_local(
                folder_path=self.index_dir,
                embeddings=self.embedding_fn,
                allow_dangerous_deserialization=True,
            )
        else:
            # Create an empty index by building from a single empty doc then clearing it
            index = index = faiss.IndexFlatL2(len(self.embedding_fn.embed_query("hello world")))

            self._vs = FAISS(
                embedding_function=self.embedding_fn,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )

            self.save()

    def add_texts(self, texts: List[str], metadatas: List[dict]) -> None:
        self._ensure_ready()
        self._vs.add_texts(texts=texts, metadatas=metadatas)

    def similarity_search_with_score(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        self._ensure_ready()
        return self._vs.similarity_search_with_score(query=query, k=k)

    def save(self) -> None:
        self._ensure_ready()
        os.makedirs(self.index_dir, exist_ok=True)
        self._vs.save_local(self.index_dir)

    def _ensure_ready(self) -> None:
        if self._vs is None:
            raise RuntimeError("FaissStore is not initialized. Call load_or_create() first.")
