"""
Retriever module for similarity search and document retrieval.

This module implements query preprocessing, similarity search,
relevance filtering, and MMR (Maximum Marginal Relevance) retrieval.
"""

import logging
import re
from typing import Dict, List, Optional

from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Retrieval parameters
DEFAULT_TOP_K = 5
RELEVANCE_THRESHOLD = 0.5  # Lowered from 0.7 for better retrieval


class Retriever:
    """Handles document retrieval with similarity search and filtering."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        top_k: int = DEFAULT_TOP_K,
        relevance_threshold: float = RELEVANCE_THRESHOLD
    ):
        """
        Initialize retriever.
        
        Args:
            vector_store: VectorStore instance
            embedding_generator: EmbeddingGenerator instance
            top_k: Number of documents to retrieve
            relevance_threshold: Minimum similarity score threshold
        """
        self.vector_store = vector_store
        self.top_k = top_k
        self.relevance_threshold = relevance_threshold
        
        # Initialize embedding generator if not provided
        if embedding_generator:
            self.embedding_generator = embedding_generator
        else:
            # Use the same embedding model configuration as was used for indexing
            import os
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("VERTEX_API_KEY")
            self.embedding_generator = EmbeddingGenerator(api_key=api_key)
    
    def preprocess_query(self, query: str) -> str:
        """
        Preprocess query for better retrieval.
        
        Args:
            query: Original query string
            
        Returns:
            Preprocessed query string
        """
        # Convert to lowercase
        query = query.lower()
        
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query)
        
        # Strip leading/trailing whitespace
        query = query.strip()
        
        return query
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_mmr: bool = False,
        mmr_diversity: float = 0.5
    ) -> List[Dict]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query string
            top_k: Number of documents to retrieve (overrides default)
            use_mmr: Whether to use Maximum Marginal Relevance
            mmr_diversity: Diversity parameter for MMR (0.0 to 1.0)
            
        Returns:
            List of retrieved documents with metadata and scores
        """
        # Preprocess query
        processed_query = self.preprocess_query(query)
        
        if not processed_query:
            logger.warning("Empty query after preprocessing")
            return []
        
        # Generate query embedding
        try:
            query_chunks = [{'text': processed_query}]
            query_chunks = self.embedding_generator.generate_embeddings(
                query_chunks,
                show_progress=False
            )
            
            if not query_chunks or 'embedding' not in query_chunks[0]:
                logger.error("Failed to generate query embedding")
                return []
            
            query_embedding = query_chunks[0]['embedding']
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            return []
        
        # Determine number of results
        n_results = top_k if top_k is not None else self.top_k
        
        # Retrieve documents
        try:
            if use_mmr:
                results = self._retrieve_with_mmr(
                    query_embedding,
                    processed_query,
                    n_results,
                    mmr_diversity
                )
            else:
                results = self.vector_store.search(
                    query_embedding,
                    n_results=n_results * 2  # Get more for filtering
                )
                results = self._format_results(results, n_results)
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
        
        # Filter by relevance threshold
        filtered_results = [
            doc for doc in results
            if doc.get('score', 0) >= self.relevance_threshold
        ]
        
        # Limit to top_k
        if len(filtered_results) > n_results:
            filtered_results = filtered_results[:n_results]
        
        logger.info(f"Retrieved {len(filtered_results)} documents for query")
        return filtered_results
    
    def _format_results(self, results: Dict, limit: int) -> List[Dict]:
        """
        Format ChromaDB results into standardized format.
        
        Args:
            results: ChromaDB query results
            limit: Maximum number of results to return
            
        Returns:
            List of formatted document dictionaries
        """
        formatted = []
        
        if not results or 'ids' not in results:
            return formatted
        
        ids = results.get('ids', [[]])[0]
        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        distances = results.get('distances', [[]])[0]
        
        for i in range(min(len(ids), limit)):
            # Convert distance to similarity score (1 - distance for cosine)
            distance = distances[i] if i < len(distances) else 1.0
            similarity_score = 1.0 - distance
            
            formatted.append({
                'id': ids[i],
                'text': documents[i] if i < len(documents) else '',
                'metadata': metadatas[i] if i < len(metadatas) else {},
                'score': similarity_score,
                'distance': distance
            })
        
        return formatted
    
    def _retrieve_with_mmr(
        self,
        query_embedding: List[float],
        query_text: str,
        n_results: int,
        diversity: float
    ) -> List[Dict]:
        """
        Retrieve documents using Maximum Marginal Relevance.
        
        Args:
            query_embedding: Query embedding vector
            query_text: Original query text
            n_results: Number of results to return
            diversity: Diversity parameter (0.0 = relevance only, 1.0 = diversity only)
            
        Returns:
            List of diverse, relevant documents
        """
        # Get initial larger set
        initial_results = self.vector_store.search(
            query_embedding,
            n_results=n_results * 3
        )
        
        candidates = self._format_results(initial_results, n_results * 3)
        
        if not candidates:
            return []
        
        # MMR selection
        selected = []
        remaining = candidates.copy()
        
        # Select first document (most relevant)
        if remaining:
            selected.append(remaining.pop(0))
        
        # Select remaining documents using MMR
        while len(selected) < n_results and remaining:
            best_score = -float('inf')
            best_idx = 0
            
            for idx, candidate in enumerate(remaining):
                # Relevance to query
                relevance = candidate['score']
                
                # Max similarity to already selected
                max_similarity = 0.0
                if selected:
                    # Calculate similarity to selected documents
                    candidate_embedding = self._get_embedding_for_text(candidate['text'])
                    if candidate_embedding:
                        for selected_doc in selected:
                            selected_embedding = self._get_embedding_for_text(selected_doc['text'])
                            if selected_embedding:
                                # Cosine similarity
                                similarity = self._cosine_similarity(
                                    candidate_embedding,
                                    selected_embedding
                                )
                                max_similarity = max(max_similarity, similarity)
                
                # MMR score
                mmr_score = (diversity * relevance) - ((1 - diversity) * max_similarity)
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            selected.append(remaining.pop(best_idx))
        
        return selected
    
    def _get_embedding_for_text(self, text: str) -> Optional[List[float]]:
        """Get embedding for text (cached if possible)."""
        try:
            chunks = [{'text': text}]
            chunks = self.embedding_generator.generate_embeddings(
                chunks,
                show_progress=False
            )
            return chunks[0].get('embedding') if chunks else None
        except Exception:
            return None
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def format_context_for_prompt(self, retrieved_docs: List[Dict]) -> str:
        """
        Format retrieved documents as context for prompt.
        
        Args:
            retrieved_docs: List of retrieved document dictionaries
            
        Returns:
            Formatted context string
        """
        if not retrieved_docs:
            return "No relevant documentation found."
        
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs, 1):
            text = doc.get('text', '')
            metadata = doc.get('metadata', {})
            source_url = metadata.get('source_url', 'Unknown')
            title = metadata.get('title', 'Untitled')
            score = doc.get('score', 0.0)
            
            context_parts.append(
                f"[Document {i}]\n"
                f"Source: {title} ({source_url})\n"
                f"Relevance Score: {score:.2f}\n"
                f"Content:\n{text}\n"
            )
        
        return "\n---\n\n".join(context_parts)


if __name__ == "__main__":
    # Test retriever
    from src.vector_store import VectorStore
    
    store = VectorStore()
    retriever = Retriever(store)
    
    test_query = "How do I create a list in Python?"
    results = retriever.retrieve(test_query)
    print(f"Retrieved {len(results)} documents")

