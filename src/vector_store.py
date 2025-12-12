"""
Vector store module using ChromaDB for persistent storage.

This module handles document storage, retrieval, and management
in the ChromaDB vector database.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Collection name
COLLECTION_NAME = "python_docs"


class VectorStore:
    """Manages ChromaDB vector store operations."""
    
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = COLLECTION_NAME
    ):
        """
        Initialize ChromaDB client and collection.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the collection
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Initialized collection: {collection_name}")
        except Exception as e:
            logger.error(f"Error initializing collection: {e}")
            raise
    
    def add_documents(
        self,
        chunks: List[Dict],
        deduplicate: bool = True
    ) -> int:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of chunk dictionaries with 'text', 'embedding', 'metadata'
            deduplicate: Whether to skip duplicate documents
            
        Returns:
            Number of documents added
        """
        if not chunks:
            logger.warning("No chunks to add")
            return 0
        
        # Prepare data for ChromaDB
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        seen_texts = set() if deduplicate else None
        
        for idx, chunk in enumerate(chunks):
            text = chunk.get('text', '')
            embedding = chunk.get('embedding')
            metadata = chunk.get('metadata', {})
            
            if not text:
                continue
            
            if not embedding:
                logger.warning(f"Skipping chunk {idx} - no embedding")
                continue
            
            # Deduplication check
            if deduplicate:
                text_hash = hash(text)
                if text_hash in seen_texts:
                    logger.debug(f"Skipping duplicate chunk {idx}")
                    continue
                seen_texts.add(text_hash)
            
            # Generate unique ID
            chunk_id = f"chunk_{idx}_{hash(text) % 1000000}"
            
            ids.append(chunk_id)
            embeddings.append(embedding)
            documents.append(text)
            metadatas.append(metadata)
        
        # Add to collection
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            logger.info(f"Added {len(ids)} documents to collection")
            return len(ids)
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    def clear_collection(self):
        """Clear all documents from the collection."""
        try:
            # Delete the collection completely
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.warning(f"Collection may not exist: {e}")
        
        # Create a fresh collection
        try:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Created fresh collection")
        except Exception as e:
            # If collection already exists, get it
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info("Using existing collection")
    
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            return {
                'collection_name': self.collection_name,
                'document_count': count,
                'persist_directory': str(self.persist_directory)
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
    
    def check_if_indexed(self) -> bool:
        """
        Check if collection has any documents indexed.
        
        Returns:
            True if collection has documents, False otherwise
        """
        try:
            count = self.collection.count()
            return count > 0
        except Exception:
            return False
    
    def get_sample_documents(self, n: int = 5) -> List[Dict]:
        """
        Get sample documents from the collection.
        
        Args:
            n: Number of samples to retrieve
            
        Returns:
            List of sample documents with metadata
        """
        try:
            results = self.collection.get(limit=n)
            
            samples = []
            for i in range(len(results['ids'])):
                samples.append({
                    'id': results['ids'][i],
                    'text': results['documents'][i],
                    'metadata': results['metadatas'][i]
                })
            
            return samples
        except Exception as e:
            logger.error(f"Error getting sample documents: {e}")
            return []
    
    def search(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where: Optional[Dict] = None
    ) -> Dict:
        """
        Search the collection with optional metadata filtering.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            where: Metadata filter dictionary
            
        Returns:
            Search results with documents, metadatas, distances, and ids
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where
            )
            return results
        except Exception as e:
            logger.error(f"Error searching collection: {e}")
            return {}


def initialize_vector_store(
    persist_directory: str = "./chroma_db",
    collection_name: str = COLLECTION_NAME
) -> VectorStore:
    """
    Initialize and return a VectorStore instance.
    
    Args:
        persist_directory: Directory to persist ChromaDB data
        collection_name: Name of the collection
        
    Returns:
        Initialized VectorStore instance
    """
    return VectorStore(persist_directory=persist_directory, collection_name=collection_name)


if __name__ == "__main__":
    # Test vector store
    store = VectorStore()
    stats = store.get_collection_stats()
    print(f"Collection stats: {stats}")
    print(f"Indexed: {store.check_if_indexed()}")

