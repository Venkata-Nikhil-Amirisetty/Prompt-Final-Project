"""
Text chunking module for splitting documents into manageable pieces.

This module implements intelligent text chunking with metadata preservation
for RAG applications.
"""

import logging
from typing import Dict, List

import tiktoken
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Chunking parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def chunk_documents(
    documents: List[Dict],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[Dict]:
    """
    Split documents into chunks with metadata preservation.
    
    Args:
        documents: List of document dictionaries with 'content', 'url', 'title'
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        List of chunk dictionaries with metadata
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    all_chunks = []
    
    for doc_idx, doc in enumerate(documents):
        content = doc.get('content', '')
        url = doc.get('url', '')
        title = doc.get('title', 'Untitled')
        
        if not content:
            logger.warning(f"Skipping empty document: {title}")
            continue
        
        # Split into chunks
        chunks = text_splitter.split_text(content)
        
        # Create chunk metadata
        for chunk_idx, chunk_text in enumerate(chunks):
            chunk_data = {
                'text': chunk_text,
                'metadata': {
                    'source_url': url,
                    'title': title,
                    'chunk_index': chunk_idx,
                    'document_index': doc_idx,
                    'total_chunks': len(chunks),
                    'chunk_size': len(chunk_text)
                }
            }
            all_chunks.append(chunk_data)
        
        logger.info(f"Split '{title}' into {len(chunks)} chunks")
    
    logger.info(f"Total chunks created: {len(all_chunks)}")
    return all_chunks


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Count tokens in text using tiktoken.
    
    Args:
        text: Text to count tokens for
        model: Model name for tokenizer
        
    Returns:
        Number of tokens
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except KeyError:
        # Fallback to cl100k_base encoding
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))


def count_chunk_tokens(chunks: List[Dict], model: str = "gpt-3.5-turbo") -> Dict:
    """
    Count tokens for all chunks and return statistics.
    
    Args:
        chunks: List of chunk dictionaries
        model: Model name for tokenizer
        
    Returns:
        Dictionary with token statistics
    """
    token_counts = []
    
    for chunk in chunks:
        text = chunk.get('text', '')
        token_count = count_tokens(text, model)
        token_counts.append(token_count)
    
    if not token_counts:
        return {
            'total_chunks': 0,
            'total_tokens': 0,
            'avg_tokens': 0,
            'min_tokens': 0,
            'max_tokens': 0
        }
    
    return {
        'total_chunks': len(token_counts),
        'total_tokens': sum(token_counts),
        'avg_tokens': sum(token_counts) / len(token_counts),
        'min_tokens': min(token_counts),
        'max_tokens': max(token_counts)
    }


if __name__ == "__main__":
    # Test chunker
    test_docs = [
        {
            'content': "This is a test document. " * 100,
            'url': 'https://example.com/test',
            'title': 'Test Document'
        }
    ]
    
    chunks = chunk_documents(test_docs)
    print(f"Created {len(chunks)} chunks")
    
    stats = count_chunk_tokens(chunks)
    print(f"Token statistics: {stats}")

