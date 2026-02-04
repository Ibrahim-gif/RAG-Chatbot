# Unit Tests for RAG System

This directory contains comprehensive unit tests for the Retrieval-Augmented Generation (RAG) system.

## Test Files Overview

### Core Module Tests

- **`test_chunking.py`** - Tests for document chunking functionality
  - Document splitting strategies (Structure-Based, Length-Based, Markdown Header-Based)
  - Chunk size and overlap validation
  - Metadata handling
  - Edge cases (empty documents, very large chunks)

- **`test_embeddings.py`** - Tests for OpenAI embedding generation
  - Embedder initialization with different models
  - Document embedding (single and batch)
  - Query embedding
  - Embedding dimension consistency
  - API key handling

- **`test_faiss_store.py`** - Tests for FAISS vector store operations
  - Index creation and loading
  - Document addition and deletion
  - Similarity search functionality
  - Index persistence
  - Edge cases (empty store, long queries)

- **`test_openai_llm.py`** - Tests for OpenAI Chat LLM wrapper
  - LLM initialization and configuration
  - Structured generation with Pydantic models
  - Free-text generation
  - Message formatting
  - Token and temperature parameters

- **`test_response_formats.py`** - Tests for Pydantic response schemas
  - Citation model validation
  - LLMResponseWithCitations structure
  - RAGRouterResponse routing decisions
  - RAGSelfEval quality assessment
  - Serialization and deserialization

- **`test_rag_pipeline.py`** - Tests for RAG orchestration
  - Document ingestion (add_to_index)
  - RAG routing logic (RAGAgent)
  - Retrieval-augmented generation (RAGGeneration)
  - Conversation history handling
  - Full pipeline integration

## Running Tests

### Run All Tests
```bash
pytest tests/
```

### Run Specific Test File
```bash
pytest tests/test_chunking.py
pytest tests/test_embeddings.py
```

### Run Specific Test Class
```bash
pytest tests/test_chunking.py::TestChunkingInitialization
```

### Run Specific Test
```bash
pytest tests/test_chunking.py::TestChunkingInitialization::test_chunking_default_initialization
```

### Run Tests with Coverage
```bash
pytest --cov=src --cov-report=html tests/
```

### Run Tests in Verbose Mode
```bash
pytest -v tests/
```

### Run Tests with Output
```bash
pytest -s tests/
```

## Test Coverage

The test suite provides comprehensive coverage of:

- ✅ **Unit Tests**: Individual module functionality
- ✅ **Integration Tests**: Component interactions
- ✅ **Edge Cases**: Boundary conditions and error handling
- ✅ **Mocking**: External API calls (OpenAI, FAISS)
- ✅ **Validation**: Pydantic model schemas

## Dependencies

The testing framework uses:
- **pytest** (>=7.0.0) - Test framework
- **pytest-cov** (>=4.0.0) - Coverage reporting
- **pytest-mock** (>=3.10.0) - Mock fixtures
- **unittest-mock** (>=1.5.0) - Mock objects

Install test dependencies:
```bash
pip install -r requirements.txt
```

## Mocking Strategy

All external API calls are mocked to avoid:
- Unnecessary API costs (OpenAI)
- Network dependencies
- Rate limiting issues
- Test flakiness

Key mocked components:
- `OpenAI` client (LLM calls)
- `OpenAIEmbeddings` (embedding generation)
- `FAISS` vector store
- File I/O operations

## Test Organization

Tests are organized by module with clear class-based grouping:

```
test_module.py
├── TestClassInitialization
├── TestClassFunctionality
├── TestClassEdgeCases
└── TestClassIntegration
```

## Writing New Tests

When adding new tests:

1. **Import Statements**
   ```python
   import pytest
   from unittest.mock import patch, MagicMock
   from src.module.submodule import MyClass
   ```

2. **Test Structure**
   ```python
   class TestMyFeature:
       def test_specific_behavior(self):
           """Clear description of what is being tested."""
           # Arrange
           mock_obj = MagicMock()
           
           # Act
           result = function_under_test(mock_obj)
           
           # Assert
           assert result == expected_value
   ```

3. **Mocking External Dependencies**
   ```python
   @patch("module.path.ExternalClass")
   def test_with_mock(self, mock_external):
       mock_instance = MagicMock()
       mock_external.return_value = mock_instance
       # Test code here
   ```

4. **Use Descriptive Names**
   - ✅ `test_chunking_respects_size_limits`
   - ❌ `test_chunking`

5. **Test One Thing**
   - Each test should verify a single behavior
   - Use multiple assertion statements only for related checks

## CI/CD Integration

Add this to your CI/CD pipeline:

```yaml
- name: Run Tests
  run: pytest tests/ --cov=src --cov-report=xml

- name: Upload Coverage
  uses: codecov/codecov-action@v3
```

## Troubleshooting

### Tests fail with "Module not found"
```bash
# Make sure you're in the project root and running:
pytest tests/
```

### Tests timeout
```bash
# Add timeout to pytest.ini:
[pytest]
timeout = 300  # 5 minutes
```

### Environment variables not set
```bash
# Create .env file for tests
OPENAI_API_KEY=test-key-123
APP_ENV=test
```

## Performance Considerations

- Unit tests should complete in <1 second each
- Total test suite should complete in <30 seconds
- Use mocks to avoid I/O operations
- Avoid creating actual files/databases in tests

## Future Improvements

- [ ] Add integration tests with real embeddings
- [ ] Add performance benchmarks
- [ ] Add API endpoint tests (test_main.py)
- [ ] Add database migration tests
- [ ] Add security/input validation tests
