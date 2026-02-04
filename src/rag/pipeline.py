"""
RAG Pipeline module.

This module orchestrates the Retrieval-Augmented Generation (RAG) pipeline.
It handles document ingestion, routing decisions, and both direct and retrieval-augmented
answer generation. Includes LangSmith observability integration.

"""

from src.chunking.basic import Chunking
from src.stores.faiss_store import FaissStore
from src.llms.openai_llm import OpenAIChatLLM
from src.embedding.openai_embeds import OpenAIEmbedder
from src.response_formats.definitions import LLMResponseWithCitations, RAGRouterResponse
from langchain_community.document_loaders import PyPDFLoader
from langsmith import traceable
import logging

@traceable(name="add_to_index", run_type="tool")
def add_to_index(file_path: str, document_type: str = "pdf", configs: dict | None = None):
    """
    Load a document, chunk it, embed it, and add it to the vector store.
    
    This function orchestrates the document ingestion pipeline:
    1. Loads the document (PDF or Markdown)
    2. Chunks it according to the specified strategy
    3. Generates embeddings for each chunk
    4. Stores the embeddings in FAISS
    
    Args:
        file_path (str): Path to the document file to ingest.
        document_type (str, optional): Type of document ('pdf' or 'md'). Defaults to "pdf".
        configs (dict, optional): Configuration dictionary with:
            - embedder_model_config.model: Embedding model name
            - Defaults to None (uses defaults from OpenAIEmbedder).
    
    Returns:
        bool: True if ingestion was successful.
    
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If document_type is not 'pdf' or 'md'.
    """
    # Load and chunk the document
    logging.info(f"Adding document {file_path} to index with document type {document_type}")
    
    #Load Document  
    if document_type == "pdf":    
        pdf_loader = PyPDFLoader(file_path=file_path)
        document = pdf_loader.load()
    elif document_type == "md":
        with open(file_path, "r", encoding="utf-8") as f:
            document = f.read()

    chunker = Chunking(chunk_size=configs["chunking_config"]["chunk_size"], chunk_overlap=configs["chunking_config"]["chunk_overlap"], document_name=file_path, document_type=document_type)
    chunks = chunker.chunk_document(document)
    print(f"Chunks: {chunks[0]}")

    # Initialize embedder and vector store
    embedder = OpenAIEmbedder(model_name=configs["embedder_model_config"]["model"])
    vector_store = FaissStore(embedding_fn=embedder._client)

    # Add chunks to the vector store
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    
    logging.info(f"Texts: {texts[0]}")
    logging.info(f"Metadatas: {metadatas[0]}")
    
    vector_store.add_documents(texts=texts, metadatas=metadatas)
    return True

@traceable(name="RAGAgent", run_type="chain")
def RAGAgent(user_query: str, conversation_history: list | None = None, configs: dict | None = None):
    """
    Main RAG agent that routes between direct LLM and retrieval-augmented generation.
    
    This function makes a routing decision:
    - If the LLM determines retrieval is NOT needed: returns a direct LLM response
    - If the LLM determines retrieval IS needed: proceeds to RAGGeneration
    
    The router uses conversation history and the user query to make an informed decision
    about whether external document context is necessary.
    
    Args:
        user_query (str): The current user question or query.
        conversation_history (list, optional): List of previous messages with 'role' and 'content'.
                                              Defaults to None (empty history).
        configs (dict, optional): Configuration dictionary with:
            - llm_model_config.model: LLM model name
            - llm_model_config.max_tokens: Maximum response tokens
            - llm_model_config.temperature: Response randomness
            - retriever_config.k: Number of top chunks to retrieve
            - templates.RAG_ROUTER_SYSTEM_PROMPT: System prompt for routing
            - templates.AI_ASSISTANT_SYSTEM_PROMPT: System prompt for generation
    
    Returns:
        Tuple[Union[str, LLMResponseWithCitations], Optional[List[dict]]]:
            - response: Either a string (direct) or LLMResponseWithCitations (RAG)
            - context_docs: Retrieved documents (None if direct answer)
    """
    
    #check if the llm can answer from the previous / available context before going to retrieval
    # Initialize LLM
    llm = OpenAIChatLLM(model_name=configs["llm_model_config"]["model"], max_tokens=configs["llm_model_config"]["max_tokens"], temperature=configs["llm_model_config"]["temperature"])
    rag_router = llm.structured_generate(messages=conversation_history, user_query=user_query, system_message=configs["templates"]["RAG_ROUTER_SYSTEM_PROMPT"], response_class=RAGRouterResponse, trace_name="Router")
    print(f"RAG Router Decision: Retrieve = {rag_router.fetch_vector_store}, Retrieval Query = {rag_router.retrieval_queries}")
    
    if rag_router.fetch_vector_store == False:
        # Generate response without retrieval
        response = llm.generate(messages=conversation_history, user_query=user_query, system_message=configs["templates"]["AI_ASSISTANT_SYSTEM_PROMPT"], trace_name="Direct Answer without Retrieval")
        return response, None
    else:
        # Proceed to retrieval-augmented generation
        return RAGGeneration(user_query=user_query, retriever_query= rag_router.retrieval_queries, k=configs["retriever_config"]["k"], conversation_history=conversation_history, llm=llm, configs=configs)
    
    
def RAGGeneration(user_query: str, retriever_query:list[str] | None, k: int = 4, conversation_history: list | None = None, llm: OpenAIChatLLM | None = None, configs: dict | None = None):
    """
    Generate a grounded answer using retrieval-augmented generation.
    
    This function:
    1. Retrieves relevant documents from the vector store based on queries
    2. Deduplicates and filters the retrieved chunks
    3. Injects the retrieved context into the prompt
    4. Generates an answer with citations
    
    Args:
        user_query (str): The original user question.
        retriever_query (list[str] | None): List of search queries optimized for retrieval.
                                            Falls back to user_query if None.
        k (int, optional): Number of top chunks to retrieve per query. Defaults to 4.
        conversation_history (list, optional): Previous conversation messages. Defaults to None.
        llm (OpenAIChatLLM, optional): The LLM instance to use for generation.
                                      Defaults to None (creates new instance).
        configs (dict, optional): Configuration dictionary (required if llm is None).
    
    Returns:
        Tuple[LLMResponseWithCitations, List[dict]]:
            - response: The answer with citations
            - context_docs: The retrieved documents used for grounding
    """
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
    response = llm.structured_generate(messages=conversation_history, user_query=user_content_with_ref_docs, system_message=configs["templates"]["AI_ASSISTANT_SYSTEM_PROMPT"], response_class=LLMResponseWithCitations, trace_name="RAG Answer with Citations")
    return response, noise_free_documents

@traceable(name="list_all_documents", run_type="tool")
def list_all_documents():
    """
    List all documents currently indexed in the vector store.
    
    Returns:
        list: A list of unique document filenames (with duplicates removed).
    """
    embedder = OpenAIEmbedder()
    db = FaissStore(embedding_fn=embedder._client)
    return list(set(str(document.metadata["source"]).replace("data\\docs\\","") for document in db._vs.docstore._dict.values()))

def filter_chunk_noise(user_content_with_ref_docs):
    """
    Deduplicate and clean up retrieved document chunks.
    
    Removes duplicate chunks (by ID) and extracts clean source information
    from the document metadata.
    
    Args:
        user_content_with_ref_docs (list): List of Document objects retrieved from vector store.
    
    Returns:
        list: List of cleaned document dictionaries with:
            - 'source': The filename (without path)
            - 'page_content': The text content
    """
    seen_ids = set()
    unique_docs = []
    
    for doc in user_content_with_ref_docs:
        if doc.id not in seen_ids:
            seen_ids.add(doc.id)
            unique_docs.append(doc)

    cleaned_docs = []
    for doc in unique_docs:
        item = {
            "source": doc.metadata.get("source").rsplit("\\", 1)[-1],
            "page_content": doc.page_content
        }
        if doc.metadata.get("page") is not None:
            item["page_number"] = doc.metadata.get("page")
        elif doc.metadata.get("Header 1") is not None:
            item["title"] = doc.metadata.get("Header 1")
        cleaned_docs.append(item)
    return cleaned_docs

@traceable(name="delete_from_vector_store", run_type="tool")
def delete_from_vector_store(file_name: str):
    """
    Delete a document and all its chunks from the vector store.
    
    Args:
        file_name (str): The name of the document file to delete.
    
    Returns:
        bool: True if deletion was successful.
    """
    embedder = OpenAIEmbedder()
    db = FaissStore(embedding_fn=embedder._client)
    db.delete(file_name)
    return True