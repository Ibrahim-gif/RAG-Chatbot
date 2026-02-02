from pydantic import BaseModel, Field
from typing import List, Annotated, Optional

class LLMResponseWithCitations(BaseModel):
    answer: Annotated[str, Field(description="The main answer to user's query")]
    sources: Annotated[List[str], Field(description="List of Documents used as references, check the metadata source field for file name")]
    
class RAGRouterResponse(BaseModel):
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