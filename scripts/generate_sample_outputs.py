"""
Script to generate sample query-response pairs for demonstration.

This script runs predefined queries through the RAG system and saves
the outputs as JSON files for documentation purposes.
"""

import json
import os
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

from src.chain import RAGChain
from src.retriever import Retriever
from src.vector_store import VectorStore

# Load environment
load_dotenv()

# Predefined queries for demonstration
SAMPLE_QUERIES = [
    "How do I create a list in Python?",
    "Explain Python decorators with an example",
    "What are the main Python data types?",
    "How do I handle exceptions in Python?",
    "What is the difference between a tuple and a list?"
]


def generate_sample_outputs(
    queries: List[str] = SAMPLE_QUERIES,
    output_dir: str = "examples/sample_outputs"
) -> List[Dict]:
    """
    Generate sample outputs for given queries.
    
    Args:
        queries: List of query strings
        output_dir: Directory to save outputs
        
    Returns:
        List of result dictionaries
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    print("Initializing RAG system...")
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("VERTEX_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY or VERTEX_API_KEY not found in environment")
    
    vector_store = VectorStore()
    retriever = Retriever(vector_store)
    chain = RAGChain(retriever, api_key=api_key)
    
    results = []
    
    print(f"Generating outputs for {len(queries)} queries...")
    
    for i, query in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] Processing: {query}")
        
        try:
            result = chain.invoke(query)
            
            # Save individual file
            filename = f"query_{i:02d}_{query[:30].replace(' ', '_').replace('?', '')}.json"
            filepath = Path(output_dir) / filename
            
            output_data = {
                'query': query,
                'answer': result.get('answer', ''),
                'sources': result.get('sources', []),
                'response_time': result.get('response_time', 0),
                'num_sources': result.get('num_sources', 0)
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"  ✓ Saved to {filepath}")
            print(f"  Response time: {result.get('response_time', 0):.2f}s")
            print(f"  Sources: {result.get('num_sources', 0)}")
            
            results.append(output_data)
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({
                'query': query,
                'error': str(e)
            })
    
    # Save combined results
    combined_file = Path(output_dir) / "all_outputs.json"
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Combined results saved to {combined_file}")
    
    # Generate summary report
    generate_summary_report(results, output_dir)
    
    return results


def generate_summary_report(results: List[Dict], output_dir: str):
    """Generate a summary report of the outputs."""
    report_path = Path(output_dir) / "summary_report.txt"
    
    total_queries = len(results)
    successful = len([r for r in results if 'error' not in r])
    failed = total_queries - successful
    
    avg_response_time = 0
    if successful > 0:
        times = [r.get('response_time', 0) for r in results if 'error' not in r]
        avg_response_time = sum(times) / len(times) if times else 0
    
    total_sources = sum(r.get('num_sources', 0) for r in results if 'error' not in r)
    avg_sources = total_sources / successful if successful > 0 else 0
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Sample Outputs Summary Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total Queries: {total_queries}\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"Failed: {failed}\n\n")
        f.write(f"Average Response Time: {avg_response_time:.2f}s\n")
        f.write(f"Average Sources per Query: {avg_sources:.1f}\n\n")
        f.write("Query Details:\n")
        f.write("-" * 50 + "\n")
        
        for i, result in enumerate(results, 1):
            f.write(f"\n{i}. Query: {result.get('query', 'N/A')}\n")
            if 'error' in result:
                f.write(f"   Status: FAILED - {result['error']}\n")
            else:
                f.write(f"   Status: SUCCESS\n")
                f.write(f"   Response Time: {result.get('response_time', 0):.2f}s\n")
                f.write(f"   Sources: {result.get('num_sources', 0)}\n")
                answer_preview = result.get('answer', '')[:100]
                f.write(f"   Answer Preview: {answer_preview}...\n")
    
    print(f"✓ Summary report saved to {report_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("Sample Outputs Generator")
    print("=" * 60)
    print()
    
    try:
        results = generate_sample_outputs()
        print(f"\n✅ Successfully generated {len(results)} sample outputs")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        exit(1)

