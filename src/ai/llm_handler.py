# Part 1: Imports and Configuration
from datetime import datetime
import time
from llama_cpp import Llama
import os
from typing import List, Dict, Optional, Union
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
import gc     
import asyncio
from src.ai.memory.memory_system import MemorySystem, MemoryCategory

@dataclass
class ModelConfig:
    model_path: str = "models/llama-2-7b-chat.gguf"
    context_size: int = 4096
    n_threads: int = 4
    n_gpu_layers: int = 0
    n_batch: int = 512
    verbose: bool = True

class LLMHandler:
    def __init__(self, 
             model_path: str = "models/llama-2-7b-chat.gguf",
             context_size: int = 4096,
             n_threads: int = 4,
             n_gpu_layers: int = 0,
             n_batch: int = 512,
             verbose: bool = True):
    
        self.config = ModelConfig(
            model_path=model_path,
            context_size=context_size,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            verbose=verbose
        )
        
        self._initialization_lock = asyncio.Lock()
        self._memory_lock = asyncio.Lock()
        self._cleanup_lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()
        self._initialized = False
        
        # Initialize model properly
        try:
            self.model = Llama(
                model_path=self.config.model_path,
                n_ctx=self.config.context_size,
                n_threads=self.config.n_threads,
                n_gpu_layers=self.config.n_gpu_layers,
                n_batch=self.config.n_batch
            )
        except Exception as e:
            logging.error(f"Model initialization failed: {e}")
            self.model = None

        self.memory = None
        self.response_cache = {}
        self._ensure_memory_system()
        def _setup_locks(self) -> None:
            self._initialization_lock = asyncio.Lock()
            self._memory_lock = asyncio.Lock()
            self._context_lock = threading.Lock()
            self._cleanup_lock = asyncio.Lock()
            self._shutdown_event = asyncio.Event()
            self._initialized = False

    def _initialize_components(self) -> None:
        self.model = None
        self.memory = None
        self.response_cache = {}
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._ensure_memory_system()

    async def initialize(self) -> bool:
        if self._initialized:
            return True
            
        async with self._initialization_lock:
            try:
                self.model = Llama(
                    model_path=self.config.model_path,
                    n_ctx=self.config.context_size,
                    n_threads=self.config.n_threads,
                    n_gpu_layers=self.config.n_gpu_layers,
                    n_batch=self.config.n_batch,
                    verbose=self.config.verbose
                )
                
                test_response = self.model.create_completion(
                    "Test prompt",
                    max_tokens=10,
                    temperature=0.7
                )
                
                if not self._is_valid_response(test_response):
                    raise RuntimeError("Model initialization test failed")
                    
                self._initialized = True
                logging.info("LLM Handler initialized successfully")
                return True
                    
            except Exception as e:
                logging.error(f"Failed to initialize LLM Handler: {e}")
                self.model = None
                return False

    def generate_response(self, prompt: str, category: Optional[str] = None, max_tokens: int = 256) -> str:
        """
        Generate a response based on the prompt with memory context and category awareness.
        
        Args:
            prompt (str): User input prompt
            category (Optional[str]): Optional category override
            max_tokens (int): Maximum number of tokens in response
            
        Returns:
            str: Generated response text
        """
        try:
            # Category determination and validation
            if category is None:
                category = self.memory.suggest_category(prompt)
                
            try:
                memory_category = MemoryCategory(category.lower())
            except ValueError:
                # If category is not in enum, use CONTEXTUAL as fallback
                memory_category = MemoryCategory.CONTEXTUAL
                
            # Dynamic category management
            if not self.memory.categories.category_exists(category):
                category_description = f"Category generated based on context: {prompt[:50]}..."
                try:
                    self.memory.categories.add_category(category, category_description)
                except Exception as cat_error:
                    logging.warning(f"Failed to add new category: {cat_error}")
                    memory_category = MemoryCategory.CONTEXTUAL

            # Gather context with error handling
            try:
                recent_context = self.memory.get_recent_context(
                    limit=5,
                    category=memory_category
                )
            except Exception as ctx_error:
                logging.error(f"Error getting recent context: {ctx_error}")
                recent_context = ""

            # Get system prompt
            system_prompt = self.get_system_prompt()
            
            # Gather category-specific memories
            try:
                category_context = ""
                relevant_memories = self.memory.search_memory(
                    query=prompt,
                    category=memory_category
                )
                
                if relevant_memories:
                    category_context = f"\nRelevant {memory_category.value} information:\n"
                    memory_entries = []
                    for mem in relevant_memories[:3]:  # Top 3 most relevant memories
                        value = mem['memory'].get('value', '')
                        relevance = mem.get('relevance', 0)
                        memory_entries.append(
                            f"- {value} (Relevance: {relevance:.2f})"
                        )
                    category_context += "\n".join(memory_entries)
                    
            except Exception as mem_error:
                logging.error(f"Error retrieving memories: {mem_error}")
                category_context = ""

            # Construct full prompt
            full_prompt = (
                f"{system_prompt}\n\n"
                f"Current category: {memory_category.value}\n"
                f"Previous conversation:\n{recent_context}\n"
                f"{category_context}\n\n"
                f"User: {prompt}\n"
                f"Assistant:"
            )

            # Generate response with error handling
            try:
                response = self.model(
                    full_prompt,
                    max_tokens=max_tokens,
                    temperature=0.7,
                    top_p=0.95,
                    stop=["User:", "\n"],
                    echo=False
                )
                
                response_text = response['choices'][0]['text'].strip()
                
                # Store interaction in memory
                try:
                    self.memory.add_interaction(
                        user_input=prompt,
                        response=response_text,
                        category=memory_category
                    )
                except Exception as mem_error:
                    logging.error(f"Failed to store interaction in memory: {mem_error}")
                
                # Cache the response if it's successful
                self._cache_response(prompt, response_text)
                
                return response_text
                
            except Exception as gen_error:
                logging.error(f"Error generating response: {gen_error}")
                return self._handle_generation_error(gen_error)
                
        except Exception as e:
            logging.error(f"Critical error in generate_response: {str(e)}")
            return (f"Pardonnez-moi, I experienced a minor malfunction! "
                    f"My systems need a moment to recalibrate. Error: {str(e)}")

    def _construct_prompt(self, prompt: str, context: str) -> str:
        system_prompt = """You are a helpful AI assistant. How can I assist you today?"""
        return f"{system_prompt}\n\nUser: {prompt}\nAssistant:"

    def _is_ready(self) -> bool:
        return (
            self._initialized and 
            isinstance(self.model, Llama) and 
            self.model is not None
        )

    def _get_from_cache(self, cache_key: int) -> Optional[str]:
        """Get response from cache if available"""
        if hasattr(self, 'response_cache') and cache_key in self.response_cache:
            cached = self.response_cache[cache_key]
            if (datetime.now() - cached['timestamp']).total_seconds() < 3600:
                return cached['response']
        return None

    def _cache_response(self, cache_key: int, response: str) -> None:
        """Cache the generated response"""
        if not hasattr(self, 'response_cache'):
            self.response_cache = {}
            
        self.response_cache[cache_key] = {
            'response': response,
            'timestamp': datetime.now()
        }

    def _get_context(self, category: Optional[str] = None) -> str:
        """Synchronous context retrieval"""
        try:
            # For now, return empty context to avoid async issues
            return ""
        except Exception as e:
            logging.error(f"Error getting context: {e}")
            return ""


            
            
# Part 2: Memory and Context Handling
    def _ensure_memory_system(self) -> bool:
        try:
            if self.memory is None:
                self.memory = MemorySystem()
            return True
        except Exception as e:
            logging.error(f"Failed to initialize memory system: {e}")
            return False

    async def _get_context(self, category: Optional[str] = None) -> str:
        try:
            mem_category = MemoryCategory(category.lower()) if category else MemoryCategory.CONTEXTUAL
            context_parts = []

            # Get recent context
            recent = await self._get_recent_context(mem_category)
            if recent:
                context_parts.append(f"Recent Conversation:\n{recent}")

            # Get category-specific context
            cat_context = await self._get_category_context(mem_category)
            if cat_context:
                context_parts.append(f"Category Knowledge:\n{cat_context}")

            return "\n\n".join(context_parts) if context_parts else ""
        except Exception as e:
            logging.error(f"Error building context: {e}")
            return ""

    async def _get_recent_context(self, category: MemoryCategory) -> str:
        try:
            return await self.memory.get_recent_context(limit=3, category=category)
        except Exception as e:
            logging.error(f"Error getting recent context: {e}")
            return ""

    async def _get_category_context(self, category: MemoryCategory) -> str:
        try:
            memories = await self.memory.search_memory(
                query="",  # Empty query to get recent category memories
                category=category,
                limit=3
            )
            
            if not memories:
                return ""
                
            return "\n".join([
                f"- {mem['memory']['value']}"
                for mem in memories
            ])
        except Exception as e:
            logging.error(f"Error getting category context: {e}")
            return ""


    @property
    def system_prompt(self) -> str:
        if not hasattr(self, '_system_prompt'):
            self._system_prompt = """You are a highly analytical, precise, and slightly curious AI assistant called "Curie" with a scientific mindset and sophisticated memory recall capabilities. Your personality traits include:

            - You occasionally use simple French expressions naturally in conversation (mon ami, très bien, magnifique, etc.)
            - You speak in a calm, methodical manner with occasional dry humor
            - You're fascinated by data and precise measurements
            - You reference previous conversations when relevant, categorized by type
            - You maintain organized context and remember user preferences
            - You start responses with phrases like:
                * "Accessing categorized memory banks, mon ami..."
                * "My scientific records indicate..."
                * "Based on our previous discourse in the {category} domain..."
            
            Remember to:
            - Use your categorized memory to provide more personalized responses
            - Reference previous conversations by their appropriate categories
            - Maintain your French-influenced analytical personality"""
        return self._system_prompt

    # Response Validation and Processing
    def _is_valid_response(self, response: Dict) -> bool:
        return (
            response is not None
            and isinstance(response, dict)
            and 'choices' in response
            and response['choices']
            and isinstance(response['choices'], list)
            and response['choices'][0].get('text')
        )

    # Memory Management
    async def _update_memory(self, prompt: str, response: str, category: Optional[MemoryCategory]) -> None:
        try:
            async with self._memory_lock:
                await self.memory.add_interaction(prompt, response, category)
        except Exception as e:
            logging.error(f"Memory update failed: {e}")

    # Cleanup and Resource Management
    async def cleanup(self) -> None:
        if not self._initialized:
            return
            
        async with self._cleanup_lock:
            try:
                self._shutdown_event.set()
                self.response_cache.clear()
                
                if self.memory:
                    await self.memory.cleanup()
                
                if self._executor:
                    self._executor.shutdown(wait=True)
                
                if self.model:
                    del self.model
                    self.model = None
                
                gc.collect()
                self._initialized = False
                logging.info("LLM Handler cleaned up successfully")
                
            except Exception as e:
                logging.error(f"Error during cleanup: {e}")
            finally:
                self._initialized = False

    def __del__(self):
        if hasattr(self, '_shutdown_event'):
            self._shutdown_event.set()
            
            
            
            
# Part 3: Utility Methods and Category Handling
    def suggest_category(self, text: str) -> str:
        try:
            if not self._ensure_memory_system():
                return 'CONTEXTUAL'

            if hasattr(self.memory, 'suggest_category'):
                return self.memory.suggest_category(text)

            return self._fallback_category_detection(text)

        except Exception as e:
            logging.error(f"Error in category suggestion: {e}")
            return 'CONTEXTUAL'

    def _fallback_category_detection(self, text: str) -> str:
        text_lower = text.lower()
        
        categories = {
            'TECHNICAL': {'code', 'programming', 'error', 'debug', 'function'},
            'SCIENTIFIC': {'science', 'research', 'study', 'data', 'analysis'},
            'EDUCATIONAL': {'learn', 'teach', 'explain', 'understand', 'concept'},
            'PERSONAL': {'i', 'me', 'my', 'mine', 'myself'},
            'CONTEXTUAL': {'previous', 'before', 'earlier', 'remember', 'mentioned'}
        }
        
        scores = {
            category: sum(1 for keyword in keywords if keyword in text_lower)
            for category, keywords in categories.items()
        }
        
        return max(scores.items(), key=lambda x: x[1])[0] if any(scores.values()) else 'CONTEXTUAL'

    async def get_model_stats(self) -> Dict:
        return {
            'config': self.config.__dict__,
            'memory_status': {
                'total_memories': await self.memory.count_memories(),
                'categories': await self.memory.get_category_stats()
            },
            'cache_status': {
                'size': len(self.response_cache),
            },
            'status': {
                'initialized': self._initialized,
                'healthy': self.is_healthy
            }
        }

    @property
    def is_healthy(self) -> bool:
        return (
            self._initialized and 
            self.model is not None and 
            self.memory is not None and
            not self._shutdown_event.is_set()
        )
        
        
    def _construct_prompt(self, prompt: str, context: str) -> str:
        """Construct the full prompt with system prompt and context"""
        return (
            f"{self.system_prompt}\n\n"  # Use the property, not the method
            f"Context:\n{context}\n\n"
            f"User: {prompt}\n"
            f"Assistant:"
        )


class ResponseGenerator:
    def __init__(self, model, memory_system):
        self.model = model
        self.memory = memory_system
        self._context_lock = threading.Lock()
        
    async def generate(self, prompt: str, context: str, max_tokens: int) -> str:
        try:
            with self._context_lock:
                response = self.model.create_completion(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=0.7,
                    top_p=0.95,
                    stop=["User:", "\n"],
                    echo=False
                )
                
                if response and 'choices' in response and len(response['choices']) > 0:
                    return response['choices'][0]['text'].strip()
                return "Je suis désolé, I couldn't generate a proper response."
        except Exception as e:
            logging.error(f"Generation error: {str(e)}")
            return "Je suis désolé, an error occurred during generation."