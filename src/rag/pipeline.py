from src.chunking.basic import Chunking
from src.stores.faiss_store import FaissStore
from src.llms.openai_llm import OpenAIChatLLM
from src.embedding.openai_embeds import OpenAIEmbedder
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
import logging

def add_to_index(file_path: str, document_type: str = "pdf"):
    # Load and chunk the document
    logging.info(f"Adding document {file_path} to index with document type {document_type}")
    #Load Document  
    if document_type == "pdf":    
        pdf_loader = PyPDFLoader("docs/sample.pdf")
        document = pdf_loader.load()
    elif document_type == "md":
        md_loader = TextLoader(file_path)
        document = md_loader.load()
        
    chunker = Chunking(document_type="pdf", chunk_strategy="Structure-Based")
    chunks = chunker.chunk_document(document)

    # Initialize embedder and vector store
    embedder = OpenAIEmbedder()
    vector_store = FaissStore(index_dir="data/index", embedding_fn=embedder)
    vector_store.load_or_create()

    # Add chunks to the vector store
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    vector_store.add_texts(texts=texts, metadatas=metadatas)
    vector_store.save()

def run_pipeline():
    pass
    