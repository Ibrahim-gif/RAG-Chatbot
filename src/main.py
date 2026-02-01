from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from src.rag.pipeline import add_to_index, list_all_documents, RAGAgent, delete_from_vector_store
from dotenv import load_dotenv

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
    return list_all_documents()

@app.post("/delete")
def delete_file(file_name: str = Body(..., embed=True)):
    return delete_from_vector_store(file_name)


@app.post("/upload")
async def add_documents(file: UploadFile = File(...)):
    """
    Upload a document and save it to disk.
    Supported formats: PDF (.pdf) and Markdown (.md) only.
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
        
    add_to_index(dest_path, document_type=ext.lstrip("."))

    return {
        "message": "File uploaded successfully",
        "saved_to": str(dest_path),
    }


@app.post("/get_response")
def get_response(conversation_history: list = Body(...)):
    print(f"Conversation History length: {len(conversation_history)}")
    
    if len(conversation_history) == 0:
        raise HTTPException(status_code=400, detail="Conversation history is empty")
    
    # Shorten the conversation history if needed
    max_history_length = 10  # Define a maximum length for the conversation history
    if len(conversation_history) > max_history_length:
        conversation_history = conversation_history[-max_history_length:]
    response, _ = RAGAgent(user_query=conversation_history[-1]["content"], conversation_history=conversation_history[:-1])
    if type(response) == str:
        return {"response": response}
    print(f"Sources: {response.sources}")
    return {"response": response.answer, "sources": response.sources}
     