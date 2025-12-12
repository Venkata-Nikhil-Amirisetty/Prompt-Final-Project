"""
Tests for the retriever module.
"""

import os
from unittest.mock import Mock, patch

import pytest

from src.retriever import Retriever
from src.vector_store import VectorStore


class TestRetriever:
    """Test cases for retriever functionality."""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        store = Mock(spec=VectorStore)
        store.collection = Mock()
        return store
    
    @pytest.fixture
    def retriever(self, mock_vector_store):
        """Create a retriever instance."""
        with patch('src.retriever.EmbeddingGenerator'):
            return Retriever(mock_vector_store)
    
    def test_similarity_search_returns_results(self, retriever, mock_vector_store):
        """Test that similarity search returns results."""
        # Mock search results
        mock_results = {
            'ids': [['id1', 'id2']],
            'documents': [['doc1', 'doc2']],
            'metadatas': [[{'title': 'Title1'}, {'title': 'Title2'}]],
            'distances': [[0.1, 0.2]]
        }
        
        mock_vector_store.search.return_value = mock_results
        
        # Mock embedding generation
        with patch.object(retriever.embedding_generator, 'generate_embeddings') as mock_embed:
            mock_embed.return_value = [{'embedding': [0.1] * 768}]
            
            results = retriever.retrieve("test query", top_k=2)
            
            assert len(results) > 0
            assert 'text' in results[0]
            assert 'score' in results[0]
    
    def test_relevance_filtering(self, retriever, mock_vector_store):
        """Test that relevance filtering works."""
        # Mock results with varying scores
        mock_results = {
            'ids': [['id1', 'id2', 'id3']],
            'documents': [['doc1', 'doc2', 'doc3']],
            'metadatas': [[{}, {}, {}]],
            'distances': [[0.1, 0.5, 0.9]]  # Scores: 0.9, 0.5, 0.1
        }
        
        mock_vector_store.search.return_value = mock_results
        
        # Set high threshold
        retriever.relevance_threshold = 0.7
        
        with patch.object(retriever.embedding_generator, 'generate_embeddings') as mock_embed:
            mock_embed.return_value = [{'embedding': [0.1] * 768}]
            
            results = retriever.retrieve("test query")
            
            # Should filter out low relevance results
            assert all(r['score'] >= retriever.relevance_threshold for r in results)
    
    def test_query_preprocessing(self, retriever):
        """Test query preprocessing."""
        test_cases = [
            ("  TEST  QUERY  ", "test query"),
            ("Test\nQuery", "test query"),
            ("TEST", "test"),
            ("  ", ""),
        ]
        
        for input_query, expected in test_cases:
            processed = retriever.preprocess_query(input_query)
            assert processed == expected
    
    def test_mmr_retrieval(self, retriever, mock_vector_store):
        """Test MMR retrieval."""
        # Mock initial results
        mock_results = {
            'ids': [['id1', 'id2', 'id3', 'id4', 'id5']],
            'documents': [['doc1', 'doc2', 'doc3', 'doc4', 'doc5']],
            'metadatas': [[{}, {}, {}, {}, {}]],
            'distances': [[0.1, 0.2, 0.3, 0.4, 0.5]]
        }
        
        mock_vector_store.search.return_value = mock_results
        
        with patch.object(retriever.embedding_generator, 'generate_embeddings') as mock_embed:
            # Mock query embedding
            mock_embed.return_value = [{'embedding': [0.1] * 768}]
            
            # Mock document embeddings for MMR
            with patch.object(retriever, '_get_embedding_for_text') as mock_doc_embed:
                mock_doc_embed.return_value = [0.1] * 768
                
                results = retriever.retrieve("test query", top_k=3, use_mmr=True)
                
                # Should return diverse results
                assert len(results) <= 3
    
    def test_empty_query_handling(self, retriever):
        """Test handling of empty queries."""
        results = retriever.retrieve("")
        assert results == []
        
        results = retriever.retrieve("   ")
        assert results == []
    
    def test_format_context_for_prompt(self, retriever):
        """Test context formatting for prompts."""
        retrieved_docs = [
            {
                'text': 'Test content 1',
                'metadata': {
                    'source_url': 'https://test.com/1',
                    'title': 'Test Title 1'
                },
                'score': 0.9
            },
            {
                'text': 'Test content 2',
                'metadata': {
                    'source_url': 'https://test.com/2',
                    'title': 'Test Title 2'
                },
                'score': 0.8
            }
        ]
        
        context = retriever.format_context_for_prompt(retrieved_docs)
        
        assert 'Test content 1' in context
        assert 'Test Title 1' in context
        assert 'https://test.com/1' in context
        assert '0.90' in context or '0.9' in context


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

