# AI Engineer Case Study: Energy Technical Documentation RAG

## Overview

This project implements a production‑ready **Retrieval‑Augmented Generation (RAG)** system with document ingestion, vector search using FAISS, LLM‑based routing, grounded answer generation with citations, and a FastAPI service layer.

The system is designed to:

* Ingest PDFs and Markdown documents
* Chunk documents using structure‑aware strategies
* Embed chunks using OpenAI embeddings
* Persist vectors locally using FAISS
* Decide dynamically (via an LLM router) whether retrieval is required
* Generate answers grounded in retrieved documents
* Expose all functionality via a REST API

---

## Project Structure

```
├── configs/             # Configuration files
├── data/docs/           # Technical documents (provided)
├── evaluation_results/  # Output from evaluation runs
├── src/                 # Code/scripts
├── tests/               # Unit and integration tests
├── .env                 # Environment file
├── README.md            # Read me file
└── requirement.txt      # python requirements file
```

---

## ⚙️ Setup & Installation

### 1. Environment Setup
Create a virtual environment to keep your dependencies isolated.

```bash
# Create the virtual environment
python -m venv .venv
```

### 2. Activate the environment
```
# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configuration
Create a .env file in the root directory and add your API keys and settings:
```
OPENAI_API_KEY=your_openai_key_here
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=[https://api.smith.langchain.com](https://api.smith.langchain.com)
LANGSMITH_API_KEY=your_langsmith_key_here
LANGSMITH_PROJECT=your_project_name
APP_ENV=dev
CONFIG_YAML_PATH=configs/prompts/prompts.yaml
CONFIG_VERSION=1.0.0
```

## 🏃 Running the Application
Start the FastAPI Server
Run the server with hot-reloading enabled for development:

```
# Retrieval‑Augmented Generation (RAG) System
uvicorn src.main:app --reload
```

Once started, you can access the API documentation at:

Swagger UI: ```http://127.0.0.1:8000/docs```

## 🧪 Testing & Evaluation
Unit Tests
Execute the test suite using pytest:

```
pytest -v /tests
```

### RAG Evaluation
To run the RAGAS evaluation pipeline and check retrieval performance:

```
python -m evaluation.ragas_eval
```

---

## Architecture

```
Client
  │
  │ REST (JSON)
  ▼
FastAPI (main.py)
  │
  ├── Document Upload / Delete / List
  └── Query Handling
          │
          ▼
     RAGAgent (pipeline.py)
          │
          ├── Router LLM (retrieve or not?)
          │
          ├── Direct LLM Answer
          │
          └── RAG Generation
          │        ├── FAISS similarity search
          │        ├── Chunk filtering & dedup
          │        └── LLM w/ citations
          ▼
       Response (answer + sources)
```

---

## Key Components

### 1. Chunking

**Location:** `basic.py`

* Supports **PDF** and **Markdown** documents
* Chunking strategies:

  * **Structure‑Based (default)** – Recursive splitting with overlap
  * **Length‑Based** – Fixed size text chunks
  * **Markdown Header‑Based** – Splits on `#`, `##`, `###`

Each chunk is stored as a LangChain `Document` with metadata including the original source filename.

---

### 2. Embeddings

**Location:** `openai_embeds.py`

* Uses OpenAI embedding model (default: `text-embedding-3-large`)
* Wraps LangChain `OpenAIEmbeddings`
* Reads API key from `OPENAI_API_KEY`

Responsibilities:

* Embed document chunks during ingestion
* Embed queries during retrieval

---

### 3. Vector Store (FAISS)

**Location:** `faiss_store.py`

* Uses FAISS `IndexFlatL2`
* Stored locally under `faiss_index/`
* Automatically loads existing index if present

Supported operations:

* `add_documents()` – persist new chunks
* `similarity_search()` – top‑k semantic retrieval
* `delete()` – remove all chunks belonging to a document

---

### 4. LLM Wrapper

**Location:** `openai_llm.py`

Encapsulates OpenAI Chat Completions with two modes:

* **generate** → free‑text responses
* **structured_generate** → Pydantic‑validated outputs

Features:

* Deterministic defaults (`temperature=0`)
* LangSmith tracing and token usage tracking
* Strict response schemas for safety and reliability

---

### 5. Structured Response Schemas

**Location:** `definitions.py`

Key schemas:

#### `RAGRouterResponse`

Used by the router LLM:

```json
{
  "fetch_vector_store": true,
  "retrieval_queries": ["azure ai authentication", "managed identity"]
}
```

#### `LLMResponseWithCitations`

Used by the RAG generator:

```json
{
  "answer": "...",
  "sources": ["doc1.pdf", "doc2.md"]
}
```

---

### 6. RAG Pipeline

**Location:** `pipeline.py`

#### Ingestion Flow

1. Load document (PDF / MD)
2. Chunk content
3. Embed chunks
4. Store in FAISS

#### Query Flow

1. Router LLM decides if retrieval is required
2. If **no retrieval** → direct LLM answer
3. If **retrieval required**:

   * Run similarity search for each retrieval query
   * Deduplicate chunks
   * Inject retrieved context into generation prompt
   * Return answer + citations

---

### 7. API Layer (FastAPI)

**Location:** `main.py`

#### Available Endpoints

##### `POST /upload`

Upload a document (PDF or Markdown)

**Request:** multipart/form‑data

**Response:**

```json
{ "message": "File uploaded and indexed successfully" }
```

---

##### `GET /files`

List all indexed documents

**Response:**

```json
["policy.pdf", "architecture.md"]
```

---

##### `POST /delete`

Delete a document from the vector store

**Request:**

```json
{ "file_name": "policy.pdf" }
```

---

##### `POST /get_response`

Get a chat response (RAG or direct)

**Request:**

```json
{
  "conversation_history": [
    {"role": "user", "content": "What is managed identity?"}
  ]
}
```

**Response (direct):**

```json
{ "response": "..." }
```

**Response (RAG):**

```json
{
  "response": "...",
  "sources": ["azure_auth.md"]
}
```

---

## Observability

* Fully instrumented with **LangSmith** tracing
* Tracks:

  * Token usage
  * Latency per LLM call
  * Model metadata
* All major pipeline steps are traceable

---

## Security & Constraints

* Only `.pdf` and `.md` files accepted
* Filenames sanitized before saving
* FAISS deserialization assumes **trusted local storage**
* API key must be provided via environment variable

---

## Known Limitations

* Local FAISS index (not distributed)
* No authentication on API endpoints
* Citation granularity is document‑level (not page/span)
* No re‑ranking stage after retrieval

---

## Typical Usage Flow

1. Start FastAPI server
2. Upload documents via `/upload`
3. Verify indexing via `/files`
4. Query system via `/get_response`
5. Receive grounded answers with citations

---

## Future Improvements

* Add re‑ranking (cross‑encoder)
* Add chunk‑level citation spans
* Add auth / rate limiting
* Move FAISS to managed vector DB
* Add self‑evaluation (`RAGSelfEval`) stage

---

**Author:** Ibrahim Shariff

**System Type:** Modular, router‑based RAG with structured LLM outputs
