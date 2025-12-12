"""
RAG chain implementation using LangChain.

This module builds the complete RAG pipeline integrating retrieval,
prompting, and LLM generation with conversation memory.
"""

import logging
import time
from typing import Dict, List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
try:
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:
    from langchain.schema import HumanMessage, SystemMessage

from src.prompts import (
    SYSTEM_PROMPT,
    format_followup_prompt,
    format_qa_prompt,
    format_conversation_history,
    NO_CONTEXT_PROMPT
)
from src.retriever import Retriever

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Chain parameters
DEFAULT_TEMPERATURE = 0.3
MAX_CONVERSATION_HISTORY = 5


class RAGChain:
    """Complete RAG chain with retrieval and generation."""
    
    def __init__(
        self,
        retriever: Retriever,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
        temperature: float = DEFAULT_TEMPERATURE
    ):
        """
        Initialize RAG chain.
        
        Args:
            retriever: Retriever instance
            api_key: Google API key for Gemini
            model_name: Gemini model name (e.g., gemini-2.5-flash, gemini-2.0-flash, gemini-pro)
            temperature: LLM temperature (0.0 to 1.0)
        """
        self.retriever = retriever
        self.temperature = temperature
        self.model_name = model_name
        self.conversation_history: List[Dict] = []
        
        # Get API key
        api_key = api_key or self._get_api_key()
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")
        
        # Initialize LLM
        # Try newer models first (gemini-2.5-flash, gemini-2.0-flash), then fallback to older ones
        model_variants = [
            model_name,  # Try requested model first
            "gemini-2.5-flash",  # Latest fast model
            "gemini-2.0-flash",  # Alternative fast model
            "gemini-pro",  # Older stable model
            "gemini-1.5-pro",  # Alternative
        ]
        # Remove duplicates while preserving order
        seen = set()
        model_variants = [m for m in model_variants if not (m in seen or seen.add(m))]
        
        last_error = None
        for model_variant in model_variants:
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model=model_variant,
                    google_api_key=api_key,
                    temperature=temperature
                )
                self.model_name = model_variant
                logger.info(f"Initialized RAG chain with Gemini model: {model_variant}")
                break
            except Exception as e:
                last_error = e
                logger.debug(f"Model {model_variant} initialization failed: {e}")
                continue
        else:
            # All variants failed during initialization
            logger.warning(f"Could not initialize model during __init__, will retry during first invoke")
            # Create with gemini-2.5-flash as default, will retry on first call if it fails
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=api_key,
                temperature=temperature
            )
            self.model_name = "gemini-2.5-flash"
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from environment."""
        import os
        from dotenv import load_dotenv
        load_dotenv()
        # Check for GOOGLE_API_KEY first, then VERTEX_API_KEY for backward compatibility
        return os.getenv("GOOGLE_API_KEY") or os.getenv("VERTEX_API_KEY")
    
    def _create_prompt_messages(
        self,
        context: str,
        question: str,
        is_followup: bool = False
    ) -> List:
        """
        Create prompt messages for LLM.
        
        Args:
            context: Retrieved documentation context
            question: User question
            is_followup: Whether this is a follow-up question
            
        Returns:
            List of message objects for LLM
        """
        if is_followup and self.conversation_history:
            history = format_conversation_history(self.conversation_history)
            prompt_text = format_followup_prompt(context, question, history)
        else:
            prompt_text = format_qa_prompt(context, question)
        
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt_text)
        ]
        
        return messages
    
    def invoke(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_mmr: bool = False,
        stream: bool = False
    ) -> Dict:
        """
        Process a query through the RAG chain.
        
        Args:
            query: User query string
            top_k: Number of documents to retrieve
            use_mmr: Whether to use MMR retrieval
            stream: Whether to stream the response
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        start_time = time.time()
        
        try:
            # Retrieve relevant documents
            retrieved_docs = self.retriever.retrieve(
                query,
                top_k=top_k,
                use_mmr=use_mmr
            )
            
            # Format context
            if retrieved_docs:
                context = self.retriever.format_context_for_prompt(retrieved_docs)
            else:
                context = ""
            
            # Determine if this is a follow-up
            is_followup = len(self.conversation_history) > 0
            
            # Generate response
            if not context or not retrieved_docs:
                answer = NO_CONTEXT_PROMPT
                sources = []
            else:
                # Create prompt messages
                messages = self._create_prompt_messages(
                    context,
                    query,
                    is_followup=is_followup
                )
                
                # Generate response
                if stream:
                    # Streaming response
                    response = self.llm.stream(messages)
                    answer = ""
                    for chunk in response:
                        if hasattr(chunk, 'content'):
                            answer += chunk.content
                else:
                    # Non-streaming response
                    try:
                        response = self.llm.invoke(messages)
                        answer = response.content if hasattr(response, 'content') else str(response)
                    except Exception as llm_error:
                        # If model error, try to reinitialize with alternative model
                        if "404" in str(llm_error) or "not found" in str(llm_error).lower():
                            logger.warning(f"Model {self.model_name} not available, trying alternative...")
                            api_key = self._get_api_key()
                            alternative_models = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-pro", "gemini-1.5-pro"]
                            if self.model_name in alternative_models:
                                alternative_models.remove(self.model_name)
                            
                            for alt_model in alternative_models:
                                try:
                                    logger.info(f"Trying alternative model: {alt_model}")
                                    self.llm = ChatGoogleGenerativeAI(
                                        model=alt_model,
                                        google_api_key=api_key,
                                        temperature=self.temperature
                                    )
                                    self.model_name = alt_model
                                    # Retry the call
                                    response = self.llm.invoke(messages)
                                    answer = response.content if hasattr(response, 'content') else str(response)
                                    logger.info(f"Successfully used model: {alt_model}")
                                    break
                                except Exception as e2:
                                    logger.warning(f"Alternative model {alt_model} also failed: {e2}")
                                    continue
                            else:
                                # All models failed - provide helpful error message
                                error_msg = (
                                    f"All Gemini models failed. The error suggests the API version (v1beta) "
                                    f"may not support these models, or the API key may be for a different service. "
                                    f"Original error: {llm_error}"
                                )
                                logger.error(error_msg)
                                raise ValueError(error_msg)
                        else:
                            raise
                
                # Extract sources
                sources = [
                    {
                        'text': doc.get('text', '')[:200] + '...',
                        'source_url': doc.get('metadata', {}).get('source_url', ''),
                        'title': doc.get('metadata', {}).get('title', 'Untitled'),
                        'score': doc.get('score', 0.0)
                    }
                    for doc in retrieved_docs
                ]
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Update conversation history
            self.conversation_history.append({
                'role': 'user',
                'content': query
            })
            self.conversation_history.append({
                'role': 'assistant',
                'content': answer
            })
            
            # Limit history size
            if len(self.conversation_history) > MAX_CONVERSATION_HISTORY * 2:
                self.conversation_history = self.conversation_history[-MAX_CONVERSATION_HISTORY * 2:]
            
            return {
                'answer': answer,
                'sources': sources,
                'response_time': response_time,
                'num_sources': len(sources),
                'query': query
            }
            
        except Exception as e:
            logger.error(f"Error in RAG chain: {e}")
            response_time = time.time() - start_time
            
            # Provide helpful error message for common issues
            error_str = str(e)
            if "404" in error_str and "v1beta" in error_str:
                user_message = (
                    "The Gemini API model is not available with your current API key. "
                    "This might happen if:\n"
                    "1. Your API key is for Vertex AI (use Vertex AI setup instead)\n"
                    "2. The model name is not supported by your API version\n"
                    "3. Your API key needs to be regenerated\n\n"
                    f"Technical error: {error_str[:200]}"
                )
            else:
                user_message = f"I encountered an error processing your query: {error_str[:200]}. Please try again."
            
            return {
                'answer': user_message,
                'sources': [],
                'response_time': response_time,
                'num_sources': 0,
                'query': query,
                'error': error_str
            }
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def update_temperature(self, temperature: float):
        """
        Update LLM temperature.
        
        Args:
            temperature: New temperature value (0.0 to 1.0)
        """
        self.temperature = temperature
        api_key = self._get_api_key()
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=api_key,
            temperature=temperature
        )
        logger.info(f"Updated temperature to {temperature}")


def create_rag_chain(
    retriever: Retriever,
    api_key: Optional[str] = None,
    temperature: float = DEFAULT_TEMPERATURE
) -> RAGChain:
    """
    Create and return a RAG chain instance.
    
    Args:
        retriever: Retriever instance
        api_key: Google API key for Gemini
        temperature: LLM temperature
        
    Returns:
        Initialized RAGChain instance
    """
    return RAGChain(
        retriever=retriever,
        api_key=api_key,
        temperature=temperature
    )


if __name__ == "__main__":
    # Test chain
    from src.vector_store import VectorStore
    from src.retriever import Retriever
    
    store = VectorStore()
    retriever = Retriever(store)
    chain = RAGChain(retriever)
    
    result = chain.invoke("How do I create a list in Python?")
    print(f"Answer: {result['answer'][:100]}...")
    print(f"Sources: {result['num_sources']}")
