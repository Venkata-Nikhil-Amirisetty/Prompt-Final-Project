"""
Script to export the knowledge base to a portable format.

This script exports the ChromaDB collection to JSON format for
easy sharing and submission.
"""

import json
from pathlib import Path
from typing import Dict, List

from src.vector_store import VectorStore


def export_knowledge_base(
    output_file: str = "outputs/knowledge_base_export.json",
    collection_name: str = "python_docs"
) -> Dict:
    """
    Export the knowledge base to JSON format.
    
    Args:
        output_file: Path to output JSON file
        collection_name: Name of the collection to export
        
    Returns:
        Dictionary containing exported data
    """
    print("Loading vector store...")
    vector_store = VectorStore(collection_name=collection_name)
    
    if not vector_store.check_if_indexed():
        print("❌ Vector store is empty. Nothing to export.")
        return {}
    
    print("Retrieving all documents...")
    
    # Get collection stats
    stats = vector_store.get_collection_stats()
    total_docs = stats.get('document_count', 0)
    
    print(f"Found {total_docs} documents")
    
    # Get all documents (in batches if needed)
    all_documents = []
    batch_size = 100
    
    # ChromaDB doesn't have a direct "get all" method, so we'll use search
    # with a dummy query to get all documents
    try:
        # Get sample to understand structure
        samples = vector_store.get_sample_documents(n=min(10, total_docs))
        
        # For full export, we'd need to iterate or use ChromaDB's get method
        # This is a simplified version that exports what we can access
        results = vector_store.collection.get()
        
        if results and 'ids' in results:
            ids = results.get('ids', [])
            documents = results.get('documents', [])
            metadatas = results.get('metadatas', [])
            
            for i in range(len(ids)):
                all_documents.append({
                    'id': ids[i],
                    'text': documents[i] if i < len(documents) else '',
                    'metadata': metadatas[i] if i < len(metadatas) else {}
                })
    except Exception as e:
        print(f"Warning: Could not retrieve all documents directly: {e}")
        print("Exporting sample documents instead...")
        all_documents = vector_store.get_sample_documents(n=100)
    
    # Create export data structure
    export_data = {
        'collection_name': collection_name,
        'total_documents': len(all_documents),
        'export_metadata': {
            'format_version': '1.0',
            'export_tool': 'export_knowledge_base.py'
        },
        'statistics': stats,
        'documents': all_documents
    }
    
    # Save to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Exported {len(all_documents)} documents to {output_path}")
    
    # Generate summary
    print("\nExport Summary:")
    print(f"  Collection: {collection_name}")
    print(f"  Documents: {len(all_documents)}")
    print(f"  Output file: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.2f} KB")
    
    return export_data


if __name__ == "__main__":
    print("=" * 60)
    print("Knowledge Base Export Tool")
    print("=" * 60)
    print()
    
    try:
        export_data = export_knowledge_base()
        print("\n✅ Export completed successfully")
    except Exception as e:
        print(f"\n❌ Error during export: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

