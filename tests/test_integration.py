"""
Integration tests for the complete RAG system.
"""

import os
import pytest
from unittest.mock import Mock, patch

from src.chain import RAGChain
from src.retriever import Retriever
from src.vector_store import VectorStore


class TestIntegration:
    """End-to-end integration tests."""
    
    @pytest.fixture
    def mock_system(self):
        """Create a mock system for testing."""
        # Mock vector store
        vector_store = Mock(spec=VectorStore)
        vector_store.check_if_indexed.return_value = True
        vector_store.get_collection_stats.return_value = {'document_count': 100}
        
        # Mock retriever
        retriever = Mock(spec=Retriever)
        retriever.retrieve.return_value = [
            {
                'text': 'Python lists are created using square brackets.',
                'metadata': {
                    'source_url': 'https://docs.python.org/3/tutorial/datastructures.html',
                    'title': 'Data Structures'
                },
                'score': 0.95
            }
        ]
        retriever.format_context_for_prompt.return_value = "Context: Python lists..."
        
        # Mock chain
        with patch('src.chain.ChatGoogleGenerativeAI') as mock_llm_class:
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.content = "You can create a list in Python using square brackets: my_list = [1, 2, 3]"
            mock_llm.invoke.return_value = mock_response
            mock_llm_class.return_value = mock_llm
            
            with patch.dict(os.environ, {'GOOGLE_API_KEY': 'test_key'}):
                chain = RAGChain(retriever, api_key='test_key')
                chain.llm = mock_llm
                
                return {
                    'vector_store': vector_store,
                    'retriever': retriever,
                    'chain': chain
                }
    
    def test_end_to_end_query(self, mock_system):
        """Test complete flow from query to response."""
        chain = mock_system['chain']
        retriever = mock_system['retriever']
        
        query = "How do I create a list in Python?"
        result = chain.invoke(query)
        
        # Verify retriever was called
        retriever.retrieve.assert_called_once()
        
        # Verify result structure
        assert 'answer' in result
        assert 'sources' in result
        assert 'response_time' in result
        assert len(result['answer']) > 0
    
    def test_multiple_queries_sequence(self, mock_system):
        """Test multiple queries in sequence."""
        chain = mock_system['chain']
        
        queries = [
            "What is a list?",
            "How do I add items?",
            "Can I have nested lists?"
        ]
        
        results = []
        for query in queries:
            result = chain.invoke(query)
            results.append(result)
        
        # Verify all queries processed
        assert len(results) == len(queries)
        
        # Verify conversation history
        assert len(chain.conversation_history) == len(queries) * 2
    
    def test_various_question_types(self, mock_system):
        """Test with various question types."""
        chain = mock_system['chain']
        
        question_types = [
            "What is...",  # Definition
            "How do I...",  # How-to
            "What is the difference between...",  # Comparison
            "Explain...",  # Explanation
            "Show me an example of...",  # Example request
        ]
        
        for question_type in question_types:
            query = f"{question_type} Python lists"
            result = chain.invoke(query)
            
            assert 'answer' in result
            assert result['answer'] is not None
    
    def test_performance_benchmark(self, mock_system):
        """Performance benchmarking test."""
        import time
        
        chain = mock_system['chain']
        query = "Test query for performance"
        
        times = []
        for _ in range(5):
            start = time.time()
            result = chain.invoke(query)
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        
        # Verify reasonable performance (mock should be fast)
        assert avg_time < 5.0  # Should be very fast with mocks
        
        # Verify all queries succeeded
        assert all('answer' in chain.invoke(query) for _ in range(3))
    
    def test_error_recovery(self, mock_system):
        """Test system recovery from errors."""
        chain = mock_system['chain']
        retriever = mock_system['retriever']
        
        # First, cause an error
        retriever.retrieve.side_effect = Exception("Temporary error")
        
        result1 = chain.invoke("Test query")
        assert 'error' in result1 or 'Error' in result1['answer']
        
        # Recover and try again
        retriever.retrieve.side_effect = None
        retriever.retrieve.return_value = [
            {
                'text': 'Recovery test',
                'metadata': {'title': 'Test'},
                'score': 0.9
            }
        ]
        
        result2 = chain.invoke("Test query")
        assert 'answer' in result2
        assert 'error' not in result2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

