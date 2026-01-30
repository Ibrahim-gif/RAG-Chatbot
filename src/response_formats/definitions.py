from pydantic import BaseModel, Field
from typing import List, Dict, Any, Annotated, Optional

class LLMResponseWithCitations(BaseModel):
    answer: Annotated[str, "The main answer to user's query"]
    citation: Annotated[List[Dict[str, Any]], "List of citations"]
    
class RAGRouterResponse(BaseModel):
    fetch_vector_store: Annotated[
        bool,
        Field(description="True if retrieval from vector store is needed, False otherwise.")
    ]
    # reason: Annotated[
    #     str,
    #     Field(description="Short justification for the decision (1-2 sentences).")
    # ]
    retrieval_query: Annotated[
        Optional[str],
        Field(
            default=None,
            description=(
                "A compact search query optimized for vector retrieval. "
                "Only set when fetch_vector_store=True. Otherwise must be null."
            ),
            min_length=3
        )
    ] = None
    