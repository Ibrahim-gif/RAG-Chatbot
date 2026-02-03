# AI Engineer Case Study: Energy Technical Documentation RAG

## Overview

Build a Retrieval-Augmented Generation (RAG) application to query energy industry technical documentation.

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

## Notes

- Document your design decisions in this README
- Add your strategy rationale
- Include evaluation results and analysis


# Retrieval‑Augmented Generation (RAG) System

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

## Configuration Requirements

The system expects a `configs` object with:

* `embedder_model_config.model`
* `llm_model_config.model`
* `llm_model_config.max_tokens`
* `llm_model_config.temperature`
* `retriever_config.k`
* `templates.RAG_ROUTER_SYSTEM_PROMPT`
* `templates.AI_ASSISTANT_SYSTEM_PROMPT`

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