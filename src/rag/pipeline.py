from src.chunking.basic import Chunking
from src.stores.faiss_store import FaissStore
from src.llms.openai_llm import OpenAIChatLLM
from src.embedding.openai_embeds import OpenAIEmbedder
from src.response_formats.definitions import LLMResponseWithCitations, RAGRouterResponse
from configs.prompts.prompts import AI_ASSISTANT_SYSTEM_PROMPT, RAG_ROUTER_SYSTEM_PROMPT
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
import logging

def add_to_index(file_path: str, document_type: str = "pdf"):
    # Load and chunk the document
    logging.info(f"Adding document {file_path} to index with document type {document_type}")
    #Load Document  
    if document_type == "pdf":    
        pdf_loader = PyPDFLoader(file_path=file_path)
        document = pdf_loader.load()
    elif document_type == "md":
        md_loader = TextLoader(file_path)
        document = md_loader.load()
        
    chunker = Chunking(document_type=document_type)
    chunks = chunker.chunk_document(document)

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

def RAGAgent(user_query: str, k: int = 5, conversation_history: list | None = None):
    
    #check if the llm can answer from the previous / available context before going to retrieval
    messages = build_conversation_history(system_prompt=RAG_ROUTER_SYSTEM_PROMPT, conversation_history=conversation_history)
    # Initialize LLM
    llm = OpenAIChatLLM(model_name="gpt-4.1-mini")
    rag_router = llm.structured_generate(messages=messages + [{"role": "user", "content": f"User Query: {user_query}"}], response_class=RAGRouterResponse)
    
    if rag_router.retrieve == False:
        # Generate response without retrieval
        response = llm.structured_generate(messages=messages + [{"role": "user", "content": user_query}], response_class=LLMResponseWithCitations)
        return response
    else:
        # Proceed to retrieval-augmented generation
        return RAGGeneration(user_query=user_query, retriever_query= rag_router.retrieval_query, k=k, conversation_history=messages, llm=llm)
    
    
def RAGGeneration(user_query: str, retriever_query:str | None, k: int = 5, conversation_history: list | None = None, llm: OpenAIChatLLM | None = None):
    # Load Vector Store
    embedder = OpenAIEmbedder()
    vector_store = FaissStore(embedding_fn=embedder)
    
    # Perform similarity search
    relevant_docs_with_scores = vector_store.similarity_search_with_score(query=retriever_query or user_query, k=k)
    relevant_docs = [doc for doc, score in relevant_docs_with_scores]
    
    user_content = {"Reference Documents": relevant_docs, "User Query": user_query}
    
    # Generate response
    response = llm.structured_generate(messages=conversation_history + {"role": "user", "content": user_content}, response_class=LLMResponseWithCitations)
    return response
    
def build_conversation_history(system_prompt: str, conversation_history: list | None):
    messages = [
            {"role": "system", "content": system_prompt},
        ]
    if conversation_history is not None:
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        for turn in conversation_history:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})
            
    return messages

def list_all_documents():
    embedder = OpenAIEmbedder()
    db = FaissStore(embedding_fn=embedder._client)
    return list(set(document.metadata["source"].replace("data\\docs\\","") for document in db._vs.docstore._dict.values()))