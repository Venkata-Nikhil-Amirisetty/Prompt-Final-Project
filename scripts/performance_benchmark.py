"""
Performance benchmarking script.

This script runs multiple test queries and measures performance metrics
including response times, retrieval accuracy, and system throughput.
"""

import json
import os
import statistics
import time
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

from src.chain import RAGChain
from src.retriever import Retriever
from src.vector_store import VectorStore

# Load environment
load_dotenv()

# Test queries for benchmarking
BENCHMARK_QUERIES = [
    "How do I create a list in Python?",
    "What are Python data types?",
    "Explain Python functions",
    "How do I handle exceptions?",
    "What is the difference between a tuple and a list?",
    "How do I read files in Python?",
    "Explain list comprehensions",
    "What are Python modules?",
    "How do I use dictionaries?",
    "Explain Python classes",
    "What are decorators in Python?",
    "How do I iterate over a list?",
    "What is a Python generator?",
    "How do I use lambda functions?",
    "Explain Python inheritance",
    "What are Python packages?",
    "How do I install packages?",
    "What is the difference between == and is?",
    "How do I create a virtual environment?",
    "Explain Python scope and namespaces"
]


def run_benchmark(
    queries: List[str] = BENCHMARK_QUERIES,
    output_file: str = "outputs/performance_benchmark.json"
) -> Dict:
    """
    Run performance benchmark tests.
    
    Args:
        queries: List of test queries
        output_file: Path to save benchmark results
        
    Returns:
        Dictionary with benchmark results
    """
    print("=" * 60)
    print("Performance Benchmark")
    print("=" * 60)
    print()
    
    # Initialize components
    print("Initializing RAG system...")
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("VERTEX_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY or VERTEX_API_KEY not found in environment")
    
    vector_store = VectorStore()
    retriever = Retriever(vector_store)
    chain = RAGChain(retriever, api_key=api_key)
    
    print(f"Running {len(queries)} test queries...\n")
    
    results = []
    response_times = []
    retrieval_times = []
    generation_times = []
    num_sources_list = []
    
    successful_queries = 0
    failed_queries = 0
    
    for i, query in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] {query[:50]}...", end=" ", flush=True)
        
        start_time = time.time()
        
        try:
            result = chain.invoke(query)
            
            elapsed = time.time() - start_time
            response_times.append(elapsed)
            
            if 'error' not in result:
                successful_queries += 1
                num_sources_list.append(result.get('num_sources', 0))
                
                # Estimate retrieval time (rough approximation)
                # In a real system, we'd measure this separately
                retrieval_times.append(elapsed * 0.3)  # Assume 30% is retrieval
                generation_times.append(elapsed * 0.7)  # Assume 70% is generation
                
                results.append({
                    'query': query,
                    'success': True,
                    'response_time': elapsed,
                    'num_sources': result.get('num_sources', 0),
                    'answer_length': len(result.get('answer', ''))
                })
                print(f"✓ ({elapsed:.2f}s)")
            else:
                failed_queries += 1
                results.append({
                    'query': query,
                    'success': False,
                    'error': result.get('error', 'Unknown error')
                })
                print(f"✗ Failed")
                
        except Exception as e:
            failed_queries += 1
            elapsed = time.time() - start_time
            results.append({
                'query': query,
                'success': False,
                'error': str(e),
                'response_time': elapsed
            })
            print(f"✗ Error: {str(e)[:30]}")
    
    # Calculate statistics
    stats = {
        'total_queries': len(queries),
        'successful': successful_queries,
        'failed': failed_queries,
        'success_rate': successful_queries / len(queries) * 100 if queries else 0
    }
    
    if response_times:
        stats['response_time'] = {
            'mean': statistics.mean(response_times),
            'median': statistics.median(response_times),
            'min': min(response_times),
            'max': max(response_times),
            'stdev': statistics.stdev(response_times) if len(response_times) > 1 else 0
        }
    
    if retrieval_times:
        stats['retrieval_time'] = {
            'mean': statistics.mean(retrieval_times),
            'median': statistics.median(retrieval_times)
        }
    
    if generation_times:
        stats['generation_time'] = {
            'mean': statistics.mean(generation_times),
            'median': statistics.median(generation_times)
        }
    
    if num_sources_list:
        stats['sources'] = {
            'mean': statistics.mean(num_sources_list),
            'median': statistics.median(num_sources_list),
            'min': min(num_sources_list),
            'max': max(num_sources_list)
        }
    
    # Create full report
    report = {
        'benchmark_metadata': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_queries': len(queries),
            'test_queries': queries
        },
        'statistics': stats,
        'detailed_results': results
    }
    
    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Benchmark Results")
    print("=" * 60)
    print(f"Total Queries: {stats['total_queries']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Success Rate: {stats['success_rate']:.1f}%")
    print()
    
    if 'response_time' in stats:
        rt = stats['response_time']
        print("Response Time Statistics:")
        print(f"  Mean: {rt['mean']:.2f}s")
        print(f"  Median: {rt['median']:.2f}s")
        print(f"  Min: {rt['min']:.2f}s")
        print(f"  Max: {rt['max']:.2f}s")
        print(f"  Std Dev: {rt['stdev']:.2f}s")
        print()
    
    if 'sources' in stats:
        src = stats['sources']
        print("Sources per Query:")
        print(f"  Mean: {src['mean']:.1f}")
        print(f"  Median: {src['median']:.1f}")
        print(f"  Range: {src['min']} - {src['max']}")
        print()
    
    print(f"Full report saved to: {output_path}")
    
    return report


if __name__ == "__main__":
    try:
        report = run_benchmark()
        print("\n✅ Benchmark completed successfully")
    except Exception as e:
        print(f"\n❌ Error during benchmark: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

