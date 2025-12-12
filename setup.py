"""
Setup script for Python Documentation Assistant.

This script initializes the project by:
1. Checking environment configuration
2. Creating necessary directories
3. Scraping documentation
4. Generating embeddings
5. Building vector store
6. Running basic tests
"""

import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import project modules
from src.scraper import scrape_python_docs, load_scraped_data
from src.chunker import chunk_documents
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore


def check_env_file() -> bool:
    """Check if .env file exists and has GOOGLE_API_KEY or VERTEX_API_KEY."""
    env_path = Path(".env")
    
    if not env_path.exists():
        print("❌ .env file not found!")
        print("📝 Please create a .env file with your GOOGLE_API_KEY")
        print("   Example: GOOGLE_API_KEY=your_api_key_here")
        print("   Or: VERTEX_API_KEY=your_api_key_here")
        return False
    
    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("VERTEX_API_KEY")
    if not api_key or api_key in ["your_api_key_here", "key"]:
        print("❌ GOOGLE_API_KEY or VERTEX_API_KEY not set in .env file!")
        print("📝 Please add your Google API key to the .env file")
        print("   Get your API key from: https://makersuite.google.com/app/apikey")
        return False
    
    print("✅ Environment configuration verified")
    return True


def create_directories():
    """Create necessary directories."""
    directories = ["data", "chroma_db", "outputs", "cache/embeddings", "logs"]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Created necessary directories")


def scrape_documentation(max_pages: int = 100, force: bool = False, include_advanced: bool = True) -> bool:
    """
    Scrape Python documentation.
    
    Args:
        max_pages: Maximum number of pages to scrape
        force: Force re-scraping even if data exists
        
    Returns:
        True if successful
    """
    data_dir = Path("data")
    all_docs_file = data_dir / "all_docs.json"
    
    if all_docs_file.exists() and not force:
        print("📚 Found existing scraped data")
        response = input("   Re-scrape documentation? (y/N): ").strip().lower()
        if response != 'y':
            print("✅ Using existing scraped data")
            return True
    
    print(f"🕷️  Scraping Python documentation (max {max_pages} pages)...")
    if include_advanced:
        print("   Including:")
        print("     • Standard Library Reference (os, json, datetime, etc.)")
        print("     • Language Reference (decorators, generators, etc.)")
        print("     • Advanced topics")
    try:
        docs = scrape_python_docs(max_pages=max_pages, include_advanced=include_advanced)
        print(f"✅ Successfully scraped {len(docs)} pages")
        print(f"   Breakdown:")
        tutorial_count = sum(1 for d in docs if 'tutorial' in d.get('url', '').lower())
        library_count = sum(1 for d in docs if 'library' in d.get('url', '').lower())
        reference_count = sum(1 for d in docs if 'reference' in d.get('url', '').lower())
        print(f"     • Tutorial: {tutorial_count} pages")
        print(f"     • Library: {library_count} pages")
        print(f"     • Reference: {reference_count} pages")
        return True
    except Exception as e:
        print(f"❌ Error scraping documentation: {e}")
        return False


def build_vector_store(force: bool = False) -> bool:
    """
    Build vector store from scraped documentation.
    
    Args:
        force: Force rebuild even if index exists
        
    Returns:
        True if successful
    """
    # Load scraped data
    print("📖 Loading scraped documentation...")
    docs = load_scraped_data()
    
    if not docs:
        print("❌ No scraped documentation found. Please run scraper first.")
        return False
    
    print(f"✅ Loaded {len(docs)} documents")
    
    # Check if vector store already exists
    vector_store = VectorStore()
    if vector_store.check_if_indexed() and not force:
        print("📊 Vector store already exists")
        response = input("   Rebuild vector store? (y/N): ").strip().lower()
        if response != 'y':
            stats = vector_store.get_collection_stats()
            print(f"✅ Using existing vector store ({stats.get('document_count', 0)} documents)")
            return True
        else:
            print("🗑️  Clearing existing vector store...")
            vector_store.clear_collection()
    
    # Chunk documents
    print("✂️  Chunking documents...")
    chunks = chunk_documents(docs)
    print(f"✅ Created {len(chunks)} chunks")
    
    # Generate embeddings
    print("🔢 Generating embeddings (this may take a while)...")
    # Use Gemini embeddings if API key is available, otherwise use local model
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("VERTEX_API_KEY")
    embedding_generator = EmbeddingGenerator(api_key=api_key)
    
    try:
        chunks_with_embeddings = embedding_generator.generate_embeddings(
            chunks,
            show_progress=True
        )
        print(f"✅ Generated embeddings for {len(chunks_with_embeddings)} chunks")
    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")
        return False
    
    # Add to vector store
    print("💾 Adding documents to vector store...")
    try:
        added_count = vector_store.add_documents(chunks_with_embeddings)
        print(f"✅ Added {added_count} documents to vector store")
        
        # Display stats
        stats = vector_store.get_collection_stats()
        print(f"📊 Vector store statistics:")
        print(f"   - Documents: {stats.get('document_count', 0)}")
        
        return True
    except Exception as e:
        print(f"❌ Error adding to vector store: {e}")
        return False


def run_basic_tests() -> bool:
    """Run basic tests to verify setup."""
    print("🧪 Running basic tests...")
    
    try:
        # Test vector store
        vector_store = VectorStore()
        if not vector_store.check_if_indexed():
            print("❌ Vector store is empty")
            return False
        
        stats = vector_store.get_collection_stats()
        print(f"✅ Vector store test passed ({stats.get('document_count', 0)} documents)")
        
        # Test retriever
        from src.retriever import Retriever
        retriever = Retriever(vector_store)
        test_query = "Python lists"
        results = retriever.retrieve(test_query, top_k=3)
        
        if results:
            print(f"✅ Retriever test passed ({len(results)} results)")
        else:
            print("⚠️  Retriever returned no results (may need more data)")
        
        # Test chain (if API key available)
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("VERTEX_API_KEY")
        if api_key:
            from src.chain import RAGChain
            chain = RAGChain(retriever, api_key=api_key)
            print("✅ Chain initialization test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def generate_setup_report():
    """Generate a setup report."""
    report_path = Path("outputs") / "setup_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write("Python Documentation Assistant - Setup Report\n")
        f.write("=" * 50 + "\n\n")
        
        # Environment
        f.write("Environment:\n")
        f.write(f"  Python: {sys.version}\n")
        f.write(f"  API Key Configured: {'Yes' if (os.getenv('GOOGLE_API_KEY') or os.getenv('VERTEX_API_KEY')) else 'No'}\n\n")
        
        # Data
        docs = load_scraped_data()
        f.write(f"Scraped Documents: {len(docs)}\n\n")
        
        # Vector Store
        vector_store = VectorStore()
        stats = vector_store.get_collection_stats()
        f.write(f"Vector Store:\n")
        f.write(f"  Documents: {stats.get('document_count', 0)}\n")
        f.write(f"  Collection: {stats.get('collection_name', 'N/A')}\n")
    
    print(f"📄 Setup report saved to {report_path}")


def main():
    """Main setup function."""
    print("=" * 60)
    print("Python Documentation Assistant - Setup")
    print("=" * 60)
    print()
    
    # Step 1: Check environment
    print("Step 1: Checking environment configuration...")
    if not check_env_file():
        print("\n❌ Setup failed. Please configure your environment.")
        sys.exit(1)
    print()
    
    # Step 2: Create directories
    print("Step 2: Creating directories...")
    create_directories()
    print()
    
    # Step 3: Scrape documentation
    print("Step 3: Scraping documentation...")
    if not scrape_documentation(max_pages=20):
        print("\n❌ Setup failed at scraping step.")
        sys.exit(1)
    print()
    
    # Step 4: Build vector store
    print("Step 4: Building vector store...")
    if not build_vector_store():
        print("\n❌ Setup failed at vector store step.")
        sys.exit(1)
    print()
    
    # Step 5: Run tests
    print("Step 5: Running basic tests...")
    if not run_basic_tests():
        print("\n⚠️  Some tests failed, but setup may still work.")
    print()
    
    # Step 6: Generate report
    print("Step 6: Generating setup report...")
    generate_setup_report()
    print()
    
    # Success message
    print("=" * 60)
    print("✅ Setup completed successfully!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Start the application: streamlit run app.py")
    print("2. Open your browser to the displayed URL")
    print("3. Start asking questions about Python!")
    print()


if __name__ == "__main__":
    main()

