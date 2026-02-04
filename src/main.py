"""
FastAPI application for Retrieval-Augmented Generation (RAG) system.

This module provides REST API endpoints for uploading documents, querying the RAG system,
listing indexed documents, and deleting documents from the vector store.

"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from src.rag.pipeline import add_to_index, list_all_documents, RAGAgent, delete_from_vector_store
from dotenv import load_dotenv
from configs.prompts.configs import configs

load_dotenv()

app = FastAPI(title="RAG API for E Source", version="0.1.0")

# Allow React dev server to call FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://192.168.2.26:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".pdf", ".md"}
ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "text/markdown",
    "text/x-markdown",
}

def safe_filename(name: str) -> str:
    # strips any path components + null bytes
    return Path(name).name.replace("\x00", "")

@app.get("/files")
def list_files():
    """
    Retrieve a list of all indexed documents in the vector store.
    
    Returns:
        list: A list of document filenames currently indexed in the system.
        
    Raises:
        HTTPException: If the vector store cannot be accessed.
    """
    try:
        return list_all_documents()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector store error: {str(e)}")

@app.post("/delete")
def delete_file(file_name: str = Body(..., embed=True)):
    """
    Delete a document from the vector store.
    
    This endpoint removes all chunks associated with the specified document
    from the FAISS vector store and saves the updated index.
    
    Args:
        file_name (str): The name of the file to delete from the vector store.
        
    Returns:
        dict: Confirmation message indicating successful deletion.
        
    Raises:
        HTTPException: If the document cannot be found or deleted.
    """
    try:
        return delete_from_vector_store(file_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def add_documents(file: UploadFile = File(...)):
    """
    Upload and index a document for RAG.
    
    This endpoint accepts PDF (.pdf) or Markdown (.md) files, saves them to disk,
    and adds them to the FAISS vector store. The document is automatically chunked
    and embedded using OpenAI embeddings.
    
    Args:
        file (UploadFile): The file to upload. Must be PDF or Markdown format.
        
    Returns:
        dict: A dictionary containing:
            - message (str): Confirmation message
            - saved_to (str): Path where the file was saved
        
    Raises:
        HTTPException 400: If no filename provided, filename is invalid,
                          file extension is not .pdf or .md, or content type is invalid.
                          
    Supported file types:
        - application/pdf (.pdf)
        - text/markdown, text/x-markdown (.md)
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    filename = safe_filename(file.filename)
    if not filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only PDF (.pdf) and Markdown (.md) files are allowed",
        )

    # MIME-type check
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid content type: {file.content_type}",
        )

    docs_dir = Path("data") / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    dest_path = docs_dir / filename

    try:
        with dest_path.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)  # 1MB
                if not chunk:
                    break
                out.write(chunk)
    finally:
        await file.close()
        
    add_to_index(dest_path, document_type=ext.lstrip("."), configs=configs)

    return {
        "message": "File uploaded successfully",
        "saved_to": str(dest_path),
    }


@app.post("/get_response")
def get_response(conversation_history: list = Body(...)):
    """
    Get a RAG-based response to a user query.
    
    This endpoint processes the conversation history and generates a response
    using either direct LLM generation or retrieval-augmented generation (RAG),
    depending on whether the router LLM determines retrieval is necessary.
    
    Args:
        conversation_history (list): A list of message dictionaries with 'role' and 'content' keys.
                                    The last message should be the current user query.
        
    Returns:
        dict: A dictionary containing:
            - response (str): The generated answer
            - sources (list, optional): List of source documents used (only for RAG responses)
        
    Raises:
        HTTPException 400: If conversation history is empty.
        
    Example request:
        {
            "conversation_history": [
                {"role": "user", "content": "What is energy efficiency?"}
            ]
        }
    """
    print(f"Conversation History length: {len(conversation_history)}")
    
    if len(conversation_history) == 0:
        raise HTTPException(status_code=400, detail="Conversation history is empty")
    
    # Shorten the conversation history if needed
    max_history_length = 10  # Define a maximum length for the conversation history
    if len(conversation_history) > max_history_length:
        conversation_history = conversation_history[-max_history_length:]
    response, _ = RAGAgent(user_query=conversation_history[-1]["content"], conversation_history=conversation_history[:-1], configs=configs)
    if type(response) == str:
        return {"response": response}
    else:
        response = response.model_dump()
    print(f"Response: {response}")
    return {"response": response["answer"], "sources": response["sources"]}