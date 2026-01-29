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

---

*Your implementation notes here...*


## FAISS VectorStore
FAISS only stores and searches embedding vectors, not the original text or metadata they represent. It returns numeric indices for the nearest vectors, but has no knowledge of documents, sources, or context. LangChain therefore uses a separate docstore (such as InMemoryDocstore) to store the actual documents and metadata, along with a mapping from FAISS vector indices to document IDs. This separation allows FAISS to remain optimized for fast vector similarity search, while LangChain can reliably retrieve and return the full documents associated with search results.