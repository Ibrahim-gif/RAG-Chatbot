"""
Response format definitions using Pydantic models.

This module defines the structured response schemas used throughout the RAG system
for type-safe LLM outputs and API responses.

"""

from pydantic import BaseModel, Field
from typing import List, Annotated, Optional

class Citation(BaseModel):
    """
    Structured representation of a citation for a document.
    
    Attributes:
        source (str): The filename of the document being cited.
        section (str): The section title or page number from which the citation is taken.
    """
    source: Annotated[str, Field(description="The filename / Source of the document being cited")]
    section: Annotated[str, Field(description="The page_number of the source if present or title of the section if present from which the citation is taken")]

class LLMResponseWithCitations(BaseModel):
    """
    Structured response format for RAG answers with source citations.
    
    Used by the LLM to return both an answer and the documents that support it,
    ensuring traceability and verifiability of the response.
    
    Attributes:
        answer (str): The main answer to the user's query, grounded in provided context.
        sources (List[str]): List of source document filenames used to generate the answer.
    """
    answer: Annotated[str, Field(description="The main answer to user's query")]
    sources: Annotated[List[Citation], Field(description="List of Documents used as references")]

class RAGRouterResponse(BaseModel):
    """
    Structured response format for the RAG router LLM decision.
    
    The router LLM uses this schema to decide whether document retrieval is needed
    and to generate optimized search queries for the vector store.
    
    Attributes:
        fetch_vector_store (bool): True if retrieval from the vector store is necessary,
                                  False if the LLM can answer from its training data or
                                  conversation history alone.
        retrieval_queries (Optional[List[str]]): List of compact search queries optimized
                                                 for vector similarity search. Only populated
                                                 when fetch_vector_store=True.
    """
    fetch_vector_store: Annotated[
        bool,
        Field(description="True if retrieval from vector store is needed, False otherwise.")
    ]

    retrieval_queries: Annotated[
        Optional[List[str]],
        Field(
            default=None,
            description=(
                "A list of compact search queries optimized for vector retrieval. Make sure to include the best keywords that are relevant to the user's question."
                "Each entry should represent a distinct topic from the user question. "
                "Only set when fetch_vector_store=True. Otherwise must be null."
            ),
            min_items=1
        )
    ] = None
    
class RAGSelfEval(BaseModel):
    """
    Structured schema for self-evaluation of RAG responses.
    
    Used for assessing the quality and correctness of generated answers,
    including grounding in provided context, hallucination detection, and
    citation quality. Can be used for automated quality assessment and improvement.
    
    Attributes:
        overall_pass (bool): True only if the answer meets all quality criteria.
        grounded_in_context (bool): Whether all claims are supported by provided documents.
        correctness_score (int): Rating from 0 (wrong) to 5 (correct).
        citation_quality_score (int): Rating from 0 (no evidence) to 5 (strong evidence).
        hallucination_present (bool): True if any claim lacks supporting evidence.
        uses_outside_knowledge (bool): True if answer relies on knowledge not in provided context.
        context_missed (bool): True if relevant information in context was not used.
        confidence (int): Self-reported confidence in this evaluation (0-100).
    """
    # --- Top-line decisions ---
    overall_pass: Annotated[
        bool,
        Field(description="True only if the answer is correct, grounded, and complete enough for user needs.")
    ]

    grounded_in_context: Annotated[
        bool,
        Field(description="Are all non-trivial factual claims supported by the provided documents/context?")
    ]
    
    correctness_score: Annotated[
        int,
        Field(ge=0, le=5, description="0=wrong, 3=mixed/uncertain, 5=correct.")
    ]

    citation_quality_score: Annotated[
        int,
        Field(ge=0, le=5, description="0=no evidence, 3=evidence exists but weak/mismatched, 5=strong evidence spans.")
    ]

    hallucination_present: Annotated[
        bool,
        Field(description="True if any claim is not supported by the provided context or is fabricated.")
    ]

    uses_outside_knowledge: Annotated[
        bool,
        Field(description="True if the answer relies on knowledge not contained in the provided context.")
    ]

    context_missed: Annotated[
        bool,
        Field(description="True if relevant info is present in the context but the answer failed to use it.")
    ]

    confidence: Annotated[
        int,
        Field(ge=0, le=100, description="Self-reported confidence in this evaluation.")
    ]