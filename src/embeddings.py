"""
Embedding generation module using Google's text-embedding-004 model.

This module handles batch processing, caching, and fallback mechanisms
for generating document embeddings.
"""

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    GEMINI_EMBEDDINGS_AVAILABLE = True
except ImportError:
    GEMINI_EMBEDDINGS_AVAILABLE = False

from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Embedding parameters
BATCH_SIZE = 100
MAX_RETRIES = 3
RETRY_DELAY = 1.0


class EmbeddingGenerator:
    """Handles embedding generation with caching and fallback."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        use_gemini: bool = True,
        use_cache: bool = True,
        cache_dir: str = "cache/embeddings"
    ):
        """
        Initialize embedding generator.
        
        Args:
            api_key: Google API key for Gemini embeddings
            use_gemini: Whether to use Gemini embeddings (fallback to sentence-transformers)
            use_cache: Whether to use embedding cache
            cache_dir: Directory for caching embeddings
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("VERTEX_API_KEY")
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding models
        self.gemini_model = None
        self.local_model = None
        
        # Try Gemini embeddings first if requested and available
        if use_gemini and GEMINI_EMBEDDINGS_AVAILABLE and self.api_key:
            try:
                self.gemini_model = GoogleGenerativeAIEmbeddings(
                    model="models/text-embedding-004",
                    google_api_key=self.api_key
                )
                logger.info("Initialized Google text-embedding-004 model")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini embeddings: {e}")
        
        # Initialize fallback model (sentence-transformers)
        try:
            self.local_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Initialized sentence-transformers embedding model")
        except Exception as e:
            logger.error(f"Failed to initialize sentence-transformers model: {e}")
            if not self.gemini_model:
                raise
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[List[float]]:
        """Load embedding from cache if available."""
        if not self.use_cache:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    return data.get('embedding')
            except Exception as e:
                logger.warning(f"Error loading cache {cache_key}: {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, embedding: List[float]):
        """Save embedding to cache."""
        if not self.use_cache:
            return
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump({'embedding': embedding}, f)
        except Exception as e:
            logger.warning(f"Error saving cache {cache_key}: {e}")
    
    def _generate_with_retry(
        self,
        texts: List[str]
    ) -> List[List[float]]:
        """
        Generate embeddings with retry logic.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        # Use Gemini if available, otherwise use local model
        model = self.gemini_model if self.gemini_model else self.local_model
        
        if not model:
            raise ValueError("No embedding model available")
        
        for attempt in range(MAX_RETRIES):
            try:
                if self.gemini_model:
                    # Use Gemini embeddings
                    embeddings = self.gemini_model.embed_documents(texts)
                else:
                    # Use sentence-transformers
                    embeddings = self.local_model.encode(texts, show_progress_bar=False)
                    embeddings = embeddings.tolist()
                return embeddings
                
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                    logger.warning(
                        f"Embedding attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"All embedding attempts failed: {e}")
                    # Fallback to local model if Gemini failed
                    if self.gemini_model and self.local_model:
                        logger.info("Falling back to sentence-transformers")
                        embeddings = self.local_model.encode(texts, show_progress_bar=False)
                        return embeddings.tolist()
                    raise
    
    def generate_embeddings(
        self,
        chunks: List[Dict],
        batch_size: int = BATCH_SIZE,
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Generate embeddings for chunks with caching and batching.
        
        Args:
            chunks: List of chunk dictionaries with 'text' key
            batch_size: Number of texts to process in each batch
            show_progress: Whether to log progress
            
        Returns:
            List of chunks with 'embedding' key added
        """
        all_embeddings = []
        total_chunks = len(chunks)
        
        # Process in batches
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            batch_texts = [chunk['text'] for chunk in batch]
            
            if show_progress:
                logger.info(f"Processing batch {i // batch_size + 1}/{(total_chunks - 1) // batch_size + 1}")
            
            # Check cache and generate embeddings
            batch_embeddings = []
            texts_to_embed = []
            indices_to_embed = []
            
            for idx, text in enumerate(batch_texts):
                cache_key = self._get_cache_key(text)
                cached_embedding = self._load_from_cache(cache_key)
                
                if cached_embedding:
                    batch_embeddings.append((idx, cached_embedding))
                else:
                    texts_to_embed.append(text)
                    indices_to_embed.append(idx)
            
            # Generate embeddings for uncached texts
            if texts_to_embed:
                try:
                    new_embeddings = self._generate_with_retry(texts_to_embed)
                    
                    # Cache and store new embeddings
                    for embed_idx, embedding in enumerate(new_embeddings):
                        original_idx = indices_to_embed[embed_idx]
                        text = texts_to_embed[embed_idx]
                        cache_key = self._get_cache_key(text)
                        self._save_to_cache(cache_key, embedding)
                        batch_embeddings.append((original_idx, embedding))
                    
                except Exception as e:
                    logger.error(f"Error generating embeddings: {e}")
                    # Fill with None for failed embeddings
                    for idx in indices_to_embed:
                        batch_embeddings.append((idx, None))
            
            # Sort by original index and add to all_embeddings
            batch_embeddings.sort(key=lambda x: x[0])
            all_embeddings.extend([emb for _, emb in batch_embeddings])
        
        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, all_embeddings):
            if embedding:
                chunk['embedding'] = embedding
            else:
                logger.warning(f"Failed to generate embedding for chunk")
        
        logger.info(f"Generated embeddings for {len([e for e in all_embeddings if e])}/{total_chunks} chunks")
        return chunks


def generate_embeddings_for_chunks(
    chunks: List[Dict],
    api_key: Optional[str] = None,
    batch_size: int = BATCH_SIZE
) -> List[Dict]:
    """
    Convenience function to generate embeddings for chunks.
    
    Args:
        chunks: List of chunk dictionaries
        api_key: Google API key for Gemini
        batch_size: Batch size for processing
        
    Returns:
        Chunks with embeddings added
    """
    generator = EmbeddingGenerator(api_key=api_key)
    return generator.generate_embeddings(chunks, batch_size=batch_size)


if __name__ == "__main__":
    # Test embeddings
    test_chunks = [
        {'text': 'This is a test chunk.'},
        {'text': 'Another test chunk with more content.'}
    ]
    
    generator = EmbeddingGenerator()
    chunks_with_embeddings = generator.generate_embeddings(test_chunks)
    print(f"Generated embeddings for {len(chunks_with_embeddings)} chunks")
