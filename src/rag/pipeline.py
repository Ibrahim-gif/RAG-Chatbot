from src.chunking.basic import Chunking
from src.stores.faiss_store import FaissStore
from src.llms.openai_llm import OpenAIChatLLM
from src.embedding.openai_embeds import OpenAIEmbedder
from src.response_formats.definitions import LLMResponseWithCitations, RAGRouterResponse
from configs.prompts.prompts import AI_ASSISTANT_SYSTEM_PROMPT, RAG_ROUTER_SYSTEM_PROMPT
from langchain_community.document_loaders import PyPDFLoader
from langsmith import traceable
import logging

@traceable(name="add_to_index", run_type="tool")
def add_to_index(file_path: str, document_type: str = "pdf"):
    # Load and chunk the document
    logging.info(f"Adding document {file_path} to index with document type {document_type}")
    
    #Load Document  
    if document_type == "pdf":    
        pdf_loader = PyPDFLoader(file_path=file_path)
        document = pdf_loader.load()
    elif document_type == "md":
        with open(file_path, "r", encoding="utf-8") as f:
            document = f.read()
        
    chunker = Chunking(document_name=file_path, document_type=document_type)
    chunks = chunker.chunk_document(document)
    print(f"Chunks: {chunks[0]}")

    # Initialize embedder and vector store
    embedder = OpenAIEmbedder()
    vector_store = FaissStore(embedding_fn=embedder._client)

    # Add chunks to the vector store
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    
    logging.info(f"Texts: {texts[0]}")
    logging.info(f"Metadatas: {metadatas[0]}")
    
    vector_store.add_documents(texts=texts, metadatas=metadatas)
    return True

@traceable(name="RAGAgent", run_type="chain")
def RAGAgent(user_query: str, k: int = 4, conversation_history: list | None = None):
    
    #check if the llm can answer from the previous / available context before going to retrieval
    # Initialize LLM
    llm = OpenAIChatLLM(model_name="gpt-4.1-mini")
    rag_router = llm.structured_generate(messages=conversation_history, user_query=user_query, system_message=RAG_ROUTER_SYSTEM_PROMPT, response_class=RAGRouterResponse, trace_name="Router")
    print(f"RAG Router Decision: Retrieve = {rag_router.fetch_vector_store}, Retrieval Query = {rag_router.retrieval_queries}")
    
    if rag_router.fetch_vector_store == False:
        # Generate response without retrieval
        response = llm.generate(messages=conversation_history, user_query=user_query, system_message=AI_ASSISTANT_SYSTEM_PROMPT, trace_name="Direct Answer without Retrieval")
        
        return response, None
    else:
        # Proceed to retrieval-augmented generation
        return RAGGeneration(user_query=user_query, retriever_query= rag_router.retrieval_queries, k=k, conversation_history=conversation_history, llm=llm)
    
    
def RAGGeneration(user_query: str, retriever_query:list[str] | None, k: int = 4, conversation_history: list | None = None, llm: OpenAIChatLLM | None = None):
    # Load Vector Store
    embedder = OpenAIEmbedder()
    vector_store = FaissStore(embedding_fn=embedder._client)
    
    # Perform similarity search
    relevant_docs = []
    for query in retriever_query:    
        relevant_docs.extend(vector_store.similarity_search(query=query or user_query, k=k))
    
    # Filter out noise from chunks
    noise_free_documents = filter_chunk_noise(relevant_docs) 
    user_content_with_ref_docs = {"Reference Documents": noise_free_documents, "User Query": user_query}
    
    # Generate response
    response = llm.structured_generate(messages=conversation_history, user_query=user_content_with_ref_docs, system_message=AI_ASSISTANT_SYSTEM_PROMPT, response_class=LLMResponseWithCitations, trace_name="RAG Answer with Citations")
    return response, noise_free_documents

@traceable(name="list_all_documents", run_type="tool")
def list_all_documents():
    embedder = OpenAIEmbedder()
    db = FaissStore(embedding_fn=embedder._client)
    return list(set(str(document.metadata["source"]).replace("data\\docs\\","") for document in db._vs.docstore._dict.values()))

def filter_chunk_noise(user_content_with_ref_docs):
    seen_ids = set()
    unique_docs = []
    
    for doc in user_content_with_ref_docs:
        if doc.id not in seen_ids:
            seen_ids.add(doc.id)
            unique_docs.append(doc)
            
    return [
        {
            "source": doc.metadata.get("source").rsplit("\\", 1)[-1],
            "page_content": doc.page_content
        }
        for doc in unique_docs
    ]

@traceable(name="delete_from_vector_store", run_type="tool")
def delete_from_vector_store(file_name: str):
    embedder = OpenAIEmbedder()
    db = FaissStore(embedding_fn=embedder._client)
    db.delete(file_name)
    return True