"""
Tests for the RAG chain module.
"""

import os
from unittest.mock import Mock, patch

import pytest

from src.chain import RAGChain
from src.retriever import Retriever
from src.vector_store import VectorStore


class TestChain:
    """Test cases for RAG chain functionality."""
    
    @pytest.fixture
    def mock_retriever(self):
        """Create a mock retriever."""
        retriever = Mock(spec=Retriever)
        retriever.retrieve.return_value = [
            {
                'text': 'Test documentation content',
                'metadata': {
                    'source_url': 'https://test.com',
                    'title': 'Test Doc'
                },
                'score': 0.9
            }
        ]
        retriever.format_context_for_prompt.return_value = "Test context"
        return retriever
    
    @pytest.fixture
    def chain(self, mock_retriever):
        """Create a chain instance with mocked LLM."""
        with patch('src.chain.ChatGoogleGenerativeAI') as mock_llm_class:
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.content = "Test response from LLM"
            mock_llm.invoke.return_value = mock_response
            mock_llm_class.return_value = mock_llm
            
            with patch.dict(os.environ, {'GOOGLE_API_KEY': 'test_key'}):
                chain = RAGChain(mock_retriever, api_key='test_key')
                chain.llm = mock_llm
                return chain
    
    def test_chain_produces_responses(self, chain, mock_retriever):
        """Test that chain produces responses."""
        result = chain.invoke("Test question")
        
        assert 'answer' in result
        assert result['answer'] == "Test response from LLM"
        assert 'sources' in result
        assert 'response_time' in result
    
    def test_conversation_memory(self, chain, mock_retriever):
        """Test that conversation memory works."""
        # First query
        result1 = chain.invoke("What is Python?")
        assert len(chain.conversation_history) == 2
        
        # Second query (follow-up)
        result2 = chain.invoke("Tell me more")
        assert len(chain.conversation_history) == 4
        
        # Check that history contains both queries
        assert chain.conversation_history[0]['content'] == "What is Python?"
        assert chain.conversation_history[2]['content'] == "Tell me more"
    
    def test_source_attribution(self, chain, mock_retriever):
        """Test that source attribution is included."""
        result = chain.invoke("Test question")
        
        assert 'sources' in result
        assert len(result['sources']) > 0
        assert 'source_url' in result['sources'][0]
        assert 'title' in result['sources'][0]
    
    def test_error_handling(self, chain, mock_retriever):
        """Test error handling for API failures."""
        # Mock LLM to raise exception
        chain.llm.invoke.side_effect = Exception("API Error")
        
        result = chain.invoke("Test question")
        
        assert 'answer' in result
        assert 'error' in result
        assert 'API Error' in result['answer'] or 'error' in result
    
    def test_response_format(self, chain, mock_retriever):
        """Test that response format is correct."""
        result = chain.invoke("Test question")
        
        required_keys = ['answer', 'sources', 'response_time', 'num_sources', 'query']
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
    
    def test_clear_history(self, chain):
        """Test clearing conversation history."""
        chain.invoke("Question 1")
        chain.invoke("Question 2")
        
        assert len(chain.conversation_history) > 0
        
        chain.clear_history()
        
        assert len(chain.conversation_history) == 0
    
    def test_temperature_update(self, chain):
        """Test updating temperature."""
        original_temp = chain.temperature
        
        chain.update_temperature(0.7)
        
        assert chain.temperature == 0.7
        assert chain.temperature != original_temp
    
    def test_no_context_handling(self, chain, mock_retriever):
        """Test handling when no context is retrieved."""
        mock_retriever.retrieve.return_value = []
        mock_retriever.format_context_for_prompt.return_value = ""
        
        result = chain.invoke("Test question")
        
        assert 'answer' in result
        assert 'sources' in result
        assert len(result['sources']) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

