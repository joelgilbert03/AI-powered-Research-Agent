import pytest
from unittest.mock import patch
from src.pinecone_ops.operations import PineconeOperations

# Mock the google.generativeai library
@pytest.fixture
def mock_google_genai():
    with patch('google.generativeai.configure'), patch('google.generativeai.embed_content') as mock_embed:
        # Return a different embedding each time to test consistency logic if needed
        mock_embed.side_effect = [{"embedding": [0.1] * 768}, {"embedding": [0.1] * 768}, {"embedding": [0.2] * 768}]
        yield mock_embed

@pytest.fixture
def pinecone_ops(mock_google_genai):
    with patch('pinecone.init'), patch('pinecone.list_indexes'), patch('pinecone.create_index'), patch('pinecone.Index'):
        return PineconeOperations(
            api_key="test_key",
            environment="test_env",
            google_api_key="test_google_api_key"
        )

def test_embedding_consistency(pinecone_ops):
    """Test embedding generation consistency"""
    text = "This is a test sentence for embedding"
    embedding1 = pinecone_ops.research_manager._generate_embedding(text)
    embedding2 = pinecone_ops.research_manager._generate_embedding(text)
    
    # Embeddings should be identical for same text
    assert embedding1 == embedding2
    assert len(embedding1) == 768  # Gemini embedding-001 dimensions

    # A different text should have a different embedding
    text3 = "This is a different sentence."
    embedding3 = pinecone_ops.research_manager._generate_embedding(text3)
    assert embedding1 != embedding3
