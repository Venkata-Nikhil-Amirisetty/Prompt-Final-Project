"""
Streamlit web application for Python Documentation Assistant.

This is the main entry point for the RAG-based documentation assistant.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv

from src.chain import RAGChain
from src.retriever import Retriever
from src.vector_store import VectorStore
from src.embeddings import EmbeddingGenerator

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Python Documentation Assistant",
    page_icon="🐍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
    .source-card {
        background-color: #f9f9f9;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        border-left: 3px solid #2a5298;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_components():
    """Initialize RAG components (cached)."""
    try:
        # Check for API key
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("VERTEX_API_KEY")
        if not api_key:
            st.error("⚠️ GOOGLE_API_KEY or VERTEX_API_KEY not found in environment. Please set it in .env file.")
            return None, None, None
        
        # Initialize components
        vector_store = VectorStore()
        # Create embedding generator with API key to ensure consistency
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("VERTEX_API_KEY")
        embedding_generator = EmbeddingGenerator(api_key=api_key, use_gemini=True)
        retriever = Retriever(vector_store, embedding_generator=embedding_generator)
        chain = RAGChain(retriever, api_key=api_key)
        
        return vector_store, retriever, chain
    except Exception as e:
        st.error(f"Error initializing components: {e}")
        return None, None, None


def initialize_session_state():
    """Initialize Streamlit session state."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'response_times' not in st.session_state:
        st.session_state.response_times = []
    if 'total_queries' not in st.session_state:
        st.session_state.total_queries = 0


def display_chat_message(role: str, content: str):
    """Display a chat message with appropriate styling."""
    if role == "user":
        with st.chat_message("user", avatar="👤"):
            st.write(content)
    else:
        with st.chat_message("assistant", avatar="🐍"):
            st.markdown(content)


def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🐍 Python Documentation Assistant</h1>
        <p style="font-size: 1.2em; margin-top: 0.5rem;">
            AI-powered assistant using RAG and Prompt Engineering
        </p>
        <p style="font-size: 0.9em; opacity: 0.9;">
            Ask questions about Python programming and get answers based on official documentation
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📚 About")
        st.markdown("""
        This is a **Technical Documentation Assistant** built using:
        - **RAG (Retrieval-Augmented Generation)**
        - **Prompt Engineering**
        - **Google Gemini API**
        - **ChromaDB Vector Store**
        """)
        
        st.header("🔧 How It Works")
        st.markdown("""
        1. **Query Processing**: Your question is processed and embedded
        2. **Document Retrieval**: Relevant Python docs are retrieved from vector store
        3. **Context Assembly**: Retrieved docs are formatted as context
        4. **Response Generation**: Gemini LLM generates answer based on context
        5. **Source Attribution**: Sources are displayed for transparency
        """)
        
        st.header("💡 Sample Questions")
        sample_questions = [
            "Explain Python decorators with an example",
            "What are Python data types?",
            "How do I handle exceptions in Python?",
            "What is the difference between a tuple and a list?",
            "How do I read and write files in Python?",
            "Explain list comprehensions with examples",
            "What are Python modules and how do I use them?"
        ]
        
        for question in sample_questions:
            if st.button(question, key=f"sample_{hash(question)}", use_container_width=True):
                st.session_state.user_input = question
        
        # Statistics
        st.header("📊 Statistics")
        vector_store, retriever, chain = initialize_components()
        
        if vector_store:
            stats = vector_store.get_collection_stats()
            st.metric("Documents Indexed", stats.get('document_count', 0))
        
        if st.session_state.response_times:
            avg_time = sum(st.session_state.response_times) / len(st.session_state.response_times)
            st.metric("Avg Response Time", f"{avg_time:.2f}s")
        
        st.metric("Total Queries", st.session_state.total_queries)
        
        # Settings
        st.header("⚙️ Settings")
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Controls randomness in responses. Lower = more deterministic."
        )
        
        num_sources = st.slider(
            "Number of Sources",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            help="Number of documentation sources to retrieve"
        )
        
        show_sources = st.checkbox("Show Sources", value=True)
        
        # Action buttons
        st.header("🛠️ Actions")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.response_times = []
            if chain:
                chain.clear_history()
            st.rerun()
        
        if st.button("🔄 Rebuild Index", use_container_width=True):
            st.info("To rebuild the index, run: python setup.py")
    
    # Initialize components
    vector_store, retriever, chain = initialize_components()
    
    if not chain:
        st.warning("⚠️ Please configure GOOGLE_API_KEY or VERTEX_API_KEY in .env file to use the assistant.")
        st.stop()
    
    # Update temperature if changed
    if hasattr(chain, 'temperature') and chain.temperature != temperature:
        chain.update_temperature(temperature)
    
    # Main chat interface
    st.header("💬 Chat")
    
    # Display chat history
    for message in st.session_state.messages:
        display_chat_message(message["role"], message["content"])
        
        # Show sources if available
        if message["role"] == "assistant" and "sources" in message and show_sources:
            with st.expander("📎 Sources", expanded=False):
                for i, source in enumerate(message.get("sources", []), 1):
                    st.markdown(f"""
                    <div class="source-card">
                        <strong>Source {i}:</strong> {source.get('title', 'Untitled')}<br>
                        <small>URL: {source.get('source_url', 'N/A')}</small><br>
                        <small>Relevance: {source.get('score', 0):.2f}</small>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Show response time
        if message["role"] == "assistant" and "response_time" in message:
            st.caption(f"⏱️ Response time: {message['response_time']:.2f}s")
    
    # User input
    user_input = st.chat_input("Ask a question about Python...")
    
    # Handle sample question clicks
    if 'user_input' in st.session_state:
        user_input = st.session_state.user_input
        del st.session_state.user_input
    
    # Process user query
    if user_input:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})
        display_chat_message("user", user_input)
        
        # Generate response
        with st.spinner("🤔 Thinking..."):
            try:
                result = chain.invoke(user_input, top_k=num_sources)
                
                answer = result.get('answer', 'No response generated.')
                sources = result.get('sources', [])
                response_time = result.get('response_time', 0)
                
                # Add assistant message to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "response_time": response_time
                })
                
                # Update statistics
                st.session_state.response_times.append(response_time)
                st.session_state.total_queries += 1
                
                # Display response
                display_chat_message("assistant", answer)
                
                # Show sources
                if show_sources and sources:
                    with st.expander("📎 Sources", expanded=False):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"""
                            <div class="source-card">
                                <strong>Source {i}:</strong> {source.get('title', 'Untitled')}<br>
                                <small>URL: <a href="{source.get('source_url', '#')}" target="_blank">{source.get('source_url', 'N/A')}</a></small><br>
                                <small>Relevance Score: {source.get('score', 0):.2f}</small>
                                <details>
                                    <summary>Preview</summary>
                                    <p style="font-size: 0.8em; color: #666;">{source.get('text', '')}</p>
                                </details>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Show response time
                st.caption(f"⏱️ Response time: {response_time:.2f}s")
                
                # Copy button
                st.button("📋 Copy Response", key=f"copy_{len(st.session_state.messages)}")
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
        
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: #666;">
        <p>
            <strong>Python Documentation Assistant</strong> v1.0.0<br>
            Built with Streamlit, LangChain, and Google Gemini<br>
            <a href="https://github.com" target="_blank">GitHub Repository</a> | 
            <a href="https://docs.python.org" target="_blank">Python Documentation</a>
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

