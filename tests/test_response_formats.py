"""
Unit tests for response format definitions.

Tests the Pydantic models for structured responses.
"""

import pytest
from pydantic import ValidationError
from src.response_formats.definitions import (
    Citation,
    LLMResponseWithCitations,
    RAGRouterResponse,
    RAGSelfEval
)


class TestCitationModel:
    """Test Citation Pydantic model."""
    
    def test_citation_valid_creation(self):
        """Test creating a valid Citation."""
        citation = Citation(
            source="document.pdf",
            section="Page 1"
        )
        assert citation.source == "document.pdf"
        assert citation.section == "Page 1"
    
    def test_citation_missing_source(self):
        """Test Citation fails without source field."""
        with pytest.raises(ValidationError):
            Citation(section="Page 1")
    
    def test_citation_missing_section(self):
        """Test Citation fails without section field."""
        with pytest.raises(ValidationError):
            Citation(source="document.pdf")
    
    def test_citation_empty_source(self):
        """Test Citation with empty source."""
        citation = Citation(
            source="",
            section="Section"
        )
        assert citation.source == ""
    
    def test_citation_special_characters(self):
        """Test Citation with special characters."""
        citation = Citation(
            source="doc_2024-01-15_v2.0.pdf",
            section="Section 2.1: Energy Efficiency (#LEDs)"
        )
        assert citation.source == "doc_2024-01-15_v2.0.pdf"
        assert "#LEDs" in citation.section


class TestLLMResponseWithCitations:
    """Test LLMResponseWithCitations Pydantic model."""
    
    def test_response_with_citations_valid(self):
        """Test creating a valid response with citations."""
        response = LLMResponseWithCitations(
            answer="LED lighting is more efficient than incandescent.",
            sources=[
                Citation(source="efficiency_guide.pdf", section="Page 5"),
                Citation(source="led_benefits.pdf", section="Section 2")
            ]
        )
        assert response.answer == "LED lighting is more efficient than incandescent."
        assert len(response.sources) == 2
        assert response.sources[0].source == "efficiency_guide.pdf"
    
    def test_response_with_single_citation(self):
        """Test response with single citation."""
        response = LLMResponseWithCitations(
            answer="Test answer",
            sources=[Citation(source="test.pdf", section="Page 1")]
        )
        assert len(response.sources) == 1
    
    def test_response_with_empty_sources(self):
        """Test response with empty sources list."""
        response = LLMResponseWithCitations(
            answer="Answer without citations",
            sources=[]
        )
        assert response.sources == []
    
    def test_response_missing_answer(self):
        """Test response fails without answer field."""
        with pytest.raises(ValidationError):
            LLMResponseWithCitations(
                sources=[Citation(source="test.pdf", section="Page 1")]
            )
    
    def test_response_missing_sources(self):
        """Test response fails without sources field."""
        with pytest.raises(ValidationError):
            LLMResponseWithCitations(
                answer="Test answer"
            )
    
    def test_response_with_empty_answer(self):
        """Test response with empty answer string."""
        response = LLMResponseWithCitations(
            answer="",
            sources=[Citation(source="test.pdf", section="Page 1")]
        )
        assert response.answer == ""
    
    def test_response_with_long_answer(self):
        """Test response with very long answer."""
        long_answer = "This is a test answer. " * 100
        response = LLMResponseWithCitations(
            answer=long_answer,
            sources=[Citation(source="test.pdf", section="Page 1")]
        )
        assert len(response.answer) > 1000
    
    def test_response_with_multiple_citations_same_source(self):
        """Test response with multiple citations from same source."""
        response = LLMResponseWithCitations(
            answer="LED and incandescent comparison",
            sources=[
                Citation(source="efficiency.pdf", section="Page 1"),
                Citation(source="efficiency.pdf", section="Page 5"),
                Citation(source="efficiency.pdf", section="Page 10")
            ]
        )
        assert len(response.sources) == 3
        assert all(c.source == "efficiency.pdf" for c in response.sources)


class TestRAGRouterResponse:
    """Test RAGRouterResponse Pydantic model."""
    
    def test_router_response_fetch_true_with_queries(self):
        """Test router response when retrieval is needed."""
        response = RAGRouterResponse(
            fetch_vector_store=True,
            retrieval_queries=["energy efficiency tips", "LED lighting benefits"]
        )
        assert response.fetch_vector_store is True
        assert len(response.retrieval_queries) == 2
        assert "energy efficiency" in response.retrieval_queries[0]
    
    def test_router_response_fetch_false_no_queries(self):
        """Test router response when retrieval is not needed."""
        response = RAGRouterResponse(
            fetch_vector_store=False,
            retrieval_queries=None
        )
        assert response.fetch_vector_store is False
        assert response.retrieval_queries is None
    
    def test_router_response_missing_fetch_vector_store(self):
        """Test router response fails without fetch_vector_store field."""
        with pytest.raises(ValidationError):
            RAGRouterResponse(
                retrieval_queries=["query"]
            )
    
    def test_router_response_single_retrieval_query(self):
        """Test router response with single retrieval query."""
        response = RAGRouterResponse(
            fetch_vector_store=True,
            retrieval_queries=["single query"]
        )
        assert len(response.retrieval_queries) == 1
    
    def test_router_response_empty_queries_with_fetch_true(self):
        """Test router response with empty query list but fetch=True."""
        with pytest.raises(ValidationError):
            RAGRouterResponse(
                fetch_vector_store=True,
                retrieval_queries=[]  # Empty list should fail (min_items=1)
            )
    
    def test_router_response_queries_none_with_fetch_true(self):
        """Test router response with None queries and fetch=True is allowed."""
        # This tests the schema: queries can be None even if fetch=True
        # (though semantically it might be odd)
        response = RAGRouterResponse(
            fetch_vector_store=True,
            retrieval_queries=None
        )
        assert response.retrieval_queries is None
    
    def test_router_response_multiple_queries(self):
        """Test router response with multiple retrieval queries."""
        response = RAGRouterResponse(
            fetch_vector_store=True,
            retrieval_queries=[
                "Query 1",
                "Query 2",
                "Query 3",
                "Query 4",
                "Query 5"
            ]
        )
        assert len(response.retrieval_queries) == 5
    
    def test_router_response_queries_with_special_chars(self):
        """Test router response queries with special characters."""
        response = RAGRouterResponse(
            fetch_vector_store=True,
            retrieval_queries=[
                "Energy efficiency: LED vs. Incandescent",
                "Cost savings @ home (2024)",
                "What's the ROI?"
            ]
        )
        assert len(response.retrieval_queries) == 3


class TestRAGSelfEval:
    """Test RAGSelfEval Pydantic model."""
    
    def test_self_eval_all_fields_valid(self):
        """Test creating valid RAGSelfEval with all fields."""
        eval_response = RAGSelfEval(
            overall_pass=True,
            grounded_in_context=True,
            correctness_score=5,
            citation_quality_score=5,
            hallucination_present=False,
            uses_outside_knowledge=False,
            context_missed=False,
            confidence=95
        )
        assert eval_response.overall_pass is True
        assert eval_response.correctness_score == 5
        assert eval_response.confidence == 95
    
    def test_self_eval_minimum_scores(self):
        """Test RAGSelfEval with minimum score values."""
        eval_response = RAGSelfEval(
            overall_pass=False,
            grounded_in_context=False,
            correctness_score=0,
            citation_quality_score=0,
            hallucination_present=True,
            uses_outside_knowledge=True,
            context_missed=True,
            confidence=0
        )
        assert eval_response.correctness_score == 0
        assert eval_response.confidence == 0
    
    def test_self_eval_medium_confidence(self):
        """Test RAGSelfEval with medium confidence."""
        eval_response = RAGSelfEval(
            overall_pass=True,
            grounded_in_context=True,
            correctness_score=3,
            citation_quality_score=3,
            hallucination_present=False,
            uses_outside_knowledge=False,
            context_missed=False,
            confidence=50
        )
        assert eval_response.confidence == 50
    
    def test_self_eval_mixed_flags(self):
        """Test RAGSelfEval with mixed boolean flags."""
        eval_response = RAGSelfEval(
            overall_pass=True,
            grounded_in_context=True,
            correctness_score=4,
            citation_quality_score=4,
            hallucination_present=False,
            uses_outside_knowledge=True,  # Mixed: uses outside knowledge
            context_missed=False,
            confidence=70
        )
        assert eval_response.uses_outside_knowledge is True
        assert eval_response.grounded_in_context is True
    
    def test_self_eval_missing_overall_pass(self):
        """Test RAGSelfEval fails without overall_pass field."""
        with pytest.raises(ValidationError):
            RAGSelfEval(
                grounded_in_context=True,
                correctness_score=5,
                citation_quality_score=5,
                hallucination_present=False,
                uses_outside_knowledge=False,
                context_missed=False,
                confidence=95
            )
    
    def test_self_eval_various_confidence_levels(self):
        """Test RAGSelfEval with various confidence levels."""
        for confidence in [0, 25, 50, 75, 100]:
            eval_response = RAGSelfEval(
                overall_pass=True,
                grounded_in_context=True,
                correctness_score=3,
                citation_quality_score=3,
                hallucination_present=False,
                uses_outside_knowledge=False,
                context_missed=False,
                confidence=confidence
            )
            assert eval_response.confidence == confidence


class TestResponseFormatIntegration:
    """Test integration between different response format models."""
    
    def test_response_and_self_eval_together(self):
        """Test using LLMResponseWithCitations and RAGSelfEval together."""
        response = LLMResponseWithCitations(
            answer="LED lighting is highly efficient",
            sources=[Citation(source="led_guide.pdf", section="Section 2")]
        )
        
        eval_response = RAGSelfEval(
            overall_pass=True,
            grounded_in_context=True,
            correctness_score=5,
            citation_quality_score=5,
            hallucination_present=False,
            uses_outside_knowledge=False,
            context_missed=False,
            confidence=90
        )
        
        assert response.answer is not None
        assert eval_response.overall_pass is True
    
    def test_router_and_response_flow(self):
        """Test typical flow: router -> generation -> eval."""
        # Step 1: Router decides to fetch
        router_response = RAGRouterResponse(
            fetch_vector_store=True,
            retrieval_queries=["LED efficiency"]
        )
        
        # Step 2: Generate response with citations
        rag_response = LLMResponseWithCitations(
            answer="LEDs are very efficient",
            sources=[Citation(source="efficiency.pdf", section="Page 1")]
        )
        
        # Step 3: Self-evaluate
        eval_response = RAGSelfEval(
            overall_pass=True,
            grounded_in_context=True,
            correctness_score=4,
            citation_quality_score=4,
            hallucination_present=False,
            uses_outside_knowledge=False,
            context_missed=False,
            confidence=85
        )
        
        assert router_response.fetch_vector_store is True
        assert rag_response.sources is not None
        assert eval_response.confidence > 0


class TestResponseFormatSerialization:
    """Test serialization of response formats."""
    
    def test_citation_dict_serialization(self):
        """Test Citation can be converted to dict."""
        citation = Citation(source="doc.pdf", section="Page 1")
        citation_dict = citation.model_dump()
        assert citation_dict["source"] == "doc.pdf"
        assert citation_dict["section"] == "Page 1"
    
    def test_response_json_serialization(self):
        """Test LLMResponseWithCitations can be serialized to JSON."""
        response = LLMResponseWithCitations(
            answer="Test answer",
            sources=[Citation(source="test.pdf", section="Page 1")]
        )
        json_str = response.model_dump_json()
        assert "Test answer" in json_str
        assert "test.pdf" in json_str
    
    def test_router_response_dict(self):
        """Test RAGRouterResponse can be converted to dict."""
        response = RAGRouterResponse(
            fetch_vector_store=True,
            retrieval_queries=["query"]
        )
        response_dict = response.model_dump()
        assert response_dict["fetch_vector_store"] is True
        assert len(response_dict["retrieval_queries"]) == 1
