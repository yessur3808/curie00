# src/ai/memory/memory_system.py

import os
import math
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
import json
import logging
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.helpers import ensure_directory_exists
from src.ai.memory.dynamic_category import DynamicMemoryCategory

class MemoryCategory(Enum):
    PERSONAL = "personal"       # User preferences, names, personal details
    TECHNICAL = "technical"     # Technical discussions and explanations
    EMOTIONAL = "emotional"     # Emotional responses and sentiments
    FACTUAL = "factual"        # General facts and information
    CONTEXTUAL = "contextual"   # Conversation context and flow
    SCIENTIFIC = "scientific"   # Scientific discussions and data
    PREFERENCE = "preference"   # User preferences and settings

class MemorySystem:
    def __init__(self, memory_file: str = "memory/conversation_history.json"):
        self.memory_file = memory_file
        self.short_term_memory: List[Dict] = []
        self.categories = DynamicMemoryCategory()
        
        # Initialize the embedding model
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.memory_embeddings = {}
        except Exception as e:
            logging.error(f"Failed to initialize embedding model: {e}")
            # Fallback to simpler similarity calculation
            self.embedding_model = None
            
        # Initialize importance threshold
        self.importance_threshold = 0.5
            
        self._initialize_category_keywords()
        self.categorized_memory: Dict[str, Dict] = {
            category: {} for category in self.categories.list_categories()
        }
        self._load_memory()
        
    def get_recent_context(self, limit: int = 5, category: Optional[MemoryCategory] = None) -> str:
        """
        Get recent conversation context, optionally filtered by category.
        
        Args:
            limit (int): Maximum number of recent interactions to include
            category (Optional[MemoryCategory]): Category to filter by
            
        Returns:
            str: Formatted context string
        """
        if category:
            recent = [m for m in self.short_term_memory[-limit:]
                     if m.get('category') == category.value]
        else:
            recent = self.short_term_memory[-limit:]

        context = []
        for memory in recent:
            context.append(f"User: {memory['user_input']}")
            context.append(f"Assistant: {memory['response']}")
            if memory.get('category'):
                context.append(f"Category: {memory['category']}")
                
        return "\n".join(context) if context else ""
    
    
    @property
    def is_healthy(self) -> bool:
        """Check if memory system is healthy"""
        return (
            hasattr(self, 'short_term_memory') and
            hasattr(self, 'categorized_memory') and
            hasattr(self, 'memory_embeddings') and
            (hasattr(self, 'embedding_model') or self.embedding_model is None)
        )

    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        return {
            'short_term_count': len(self.short_term_memory),
            'categorized_count': sum(len(cat) for cat in self.categorized_memory.values()),
            'embeddings_count': len(self.memory_embeddings),
            'has_embedding_model': self.embedding_model is not None,
            'categories': list(self.categorized_memory.keys())
        }
    
    async def cleanup(self) -> None:
        """Clean up memory system resources"""
        try:
            # Save final state
            self._save_memory()
            
            # Clear memory structures
            self.short_term_memory.clear()
            self.categorized_memory.clear()
            self.memory_embeddings.clear()
            
            # Clear embedding model
            if hasattr(self, 'embedding_model'):
                del self.embedding_model
                self.embedding_model = None
            
            # Force garbage collection for good measure
            gc.collect()
            
            logging.info("Memory system cleaned up successfully")
            
        except Exception as e:
            logging.error(f"Error during memory system cleanup: {e}")

    def _embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for text with fallback"""
        try:
            if self.embedding_model is not None:
                return self.embedding_model.encode([text])[0]
            else:
                words = set(text.lower().split())
                return np.array([1 if word in words else 0 
                               for word in self.get_vocabulary()])
        except Exception as e:
            logging.error(f"Embedding generation failed: {e}")
            return np.zeros(384)  
        
    def get_vocabulary(self) -> List[str]:
        """Get or create vocabulary for fallback embedding"""
        if not hasattr(self, '_vocabulary'):
            # Create vocabulary from existing memories
            words = set()
            for memory in self.short_term_memory:
                words.update(memory['user_input'].lower().split())
                words.update(memory['response'].lower().split())
            self._vocabulary = sorted(list(words))
        return self._vocabulary

    def _calculate_importance(self, text: str, context: str) -> float:
        """Calculate importance score based on context similarity"""
        if not context:
            return 0.5
        text_embedding = self._embed_text(text)
        context_embedding = self._embed_text(context)
        return float(cosine_similarity([text_embedding], [context_embedding])[0][0])

    def _load_memory(self):
        ensure_directory_exists(os.path.dirname(self.memory_file))
        try:
            with open(self.memory_file, 'r') as f:
                data = json.load(f)
                self.categorized_memory = data.get('categorized', {
                    category.value: {} for category in MemoryCategory
                })
                self.short_term_memory = data.get('short_term', [])
                self.memory_embeddings = {
                    k: np.array(v) for k, v in data.get('embeddings', {}).items()
                }
        except FileNotFoundError:
            self._save_memory()

    def _save_memory(self):
        with open(self.memory_file, 'w') as f:
            json.dump({
                'categorized': self.categorized_memory,
                'short_term': self.short_term_memory,
                'embeddings': {
                    k: v.tolist() for k, v in self.memory_embeddings.items()
                }
            }, f, indent=2)
            
    def _initialize_category_keywords(self):
        """Initialize weighted keywords for each category"""
        self.category_keywords = {
            "PERSONAL": {
                'high_weight': {
                    'name', 'my', 'i am', 'myself', 'personal', 'family', 
                    'friend', 'relationship', 'birthday', 'profile'
                },
                'medium_weight': {
                    'like', 'prefer', 'feel', 'think', 'believe', 'remember',
                    'experience', 'background', 'history', 'life'
                },
                'context_pairs': [
                    ('my', 'is'), ('i', 'have'), ('my', 'name'),
                    ('i', 'am'), ('my', 'family')
                ]
            },
            "TECHNICAL": {
                'high_weight': {
                    'code', 'programming', 'software', 'hardware', 'system',
                    'algorithm', 'database', 'framework', 'api', 'function'
                },
                'medium_weight': {
                    'build', 'create', 'develop', 'implement', 'design',
                    'configure', 'setup', 'install', 'deploy', 'optimize'
                },
                'context_pairs': [
                    ('how', 'work'), ('how', 'build'), ('how', 'code'),
                    ('technical', 'question'), ('programming', 'language')
                ]
            },
            "EMOTIONAL": {
                'high_weight': {
                    'feel', 'emotion', 'happy', 'sad', 'angry', 'anxiety',
                    'love', 'hate', 'stress', 'mood'
                },
                'medium_weight': {
                    'tired', 'excited', 'worried', 'scared', 'proud',
                    'confident', 'nervous', 'comfortable', 'upset', 'peaceful'
                },
                'context_pairs': [
                    ('feel', 'like'), ('am', 'feeling'), ('makes', 'feel'),
                    ('emotional', 'support'), ('mood', 'today')
                ]
            },
            "FACTUAL": {
                'high_weight': {
                    'what', 'when', 'where', 'fact', 'define', 'explain',
                    'describe', 'information', 'data', 'statistics'
                },
                'medium_weight': {
                    'true', 'false', 'correct', 'incorrect', 'accurate',
                    'inaccurate', 'real', 'fake', 'authentic', 'verified'
                },
                'context_pairs': [
                    ('tell', 'about'), ('what', 'is'), ('where', 'is'),
                    ('when', 'did'), ('fact', 'check')
                ]
            },
            "SCIENTIFIC": {
                'high_weight': {
                    'science', 'research', 'study', 'experiment', 'theory',
                    'hypothesis', 'analysis', 'data', 'evidence', 'method'
                },
                'medium_weight': {
                    'test', 'measure', 'observe', 'calculate', 'predict',
                    'prove', 'demonstrate', 'investigate', 'examine', 'evaluate'
                },
                'context_pairs': [
                    ('scientific', 'method'), ('research', 'shows'),
                    ('study', 'finds'), ('data', 'analysis'), ('experimental', 'results')
                ]
            },
            "PREFERENCE": {
                'high_weight': {
                    'prefer', 'like', 'want', 'favorite', 'choice',
                    'option', 'select', 'choose', 'rather', 'better'
                },
                'medium_weight': {
                    'enjoy', 'appreciate', 'value', 'interest', 'desire',
                    'wish', 'hope', 'ideal', 'perfect', 'best'
                },
                'context_pairs': [
                    ('i', 'prefer'), ('i', 'like'), ('i', 'want'),
                    ('my', 'favorite'), ('would', 'rather')
                ]
            },
            "CONTEXTUAL": {
                'high_weight': {
                    'context', 'situation', 'scenario', 'case', 'condition',
                    'circumstance', 'environment', 'setting', 'background', 'state'
                },
                'medium_weight': {
                    'during', 'while', 'when', 'where', 'current',
                    'present', 'moment', 'now', 'here', 'there'
                },
                'context_pairs': [
                    ('in', 'this'), ('at', 'this'), ('during', 'the'),
                    ('while', 'the'), ('given', 'the')
                ]
            }
        }

    def suggest_category(self, text: str) -> str:
        """Suggest a category based on sophisticated text analysis"""
        text = text.lower()
        words = text.split()
        word_pairs = list(zip(words[:-1], words[1:]))
        
        scores = {category: 0.0 for category in self.category_keywords.keys()}
        
        for category, keywords in self.category_keywords.items():
            # High weight keywords (weight: 3.0)
            for keyword in keywords['high_weight']:
                if keyword in text:
                    scores[category] += 3.0
                    
            # Medium weight keywords (weight: 1.5)
            for keyword in keywords['medium_weight']:
                if keyword in text:
                    scores[category] += 1.5
                    
            # Context pairs (weight: 2.0)
            for pair in keywords['context_pairs']:
                if pair[0] in words and pair[1] in words:
                    word_distance = self._calculate_word_distance(words, pair[0], pair[1])
                    if word_distance <= 3:  # Words are close to each other
                        scores[category] += 2.0 * (1 / (word_distance + 1))

            # Length normalization
            scores[category] = scores[category] / math.sqrt(len(words))
            
            # Consider previous interactions
            history_bonus = self._calculate_history_bonus(category, text)
            scores[category] += history_bonus

        # Add confidence threshold
        max_score = max(scores.values())
        if max_score < 1.0:  # If no strong category is found
            return "CONTEXTUAL"
            
        return max(scores.items(), key=lambda x: x[1])[0]
    
    
    
    def suggest_category_for_llm(self, text: str) -> Optional[str]:
        """Bridge method for LLM category suggestion"""
        try:
            # First try using the sophisticated category system
            category = self.suggest_category(text)
            
            # Convert to MemoryCategory enum value if it's a valid category
            try:
                return MemoryCategory[category].value
            except KeyError:
                # If category is not in enum, use the dynamic category system
                if self.categories.category_exists(category):
                    return category
                    
                # Fallback to contextual
                return MemoryCategory.CONTEXTUAL.value
                
        except Exception as e:
            logging.error(f"Error in category suggestion for LLM: {e}")
            return MemoryCategory.CONTEXTUAL.value

    def get_category_context(self, category: str) -> str:
        """Get context specific to a category for LLM"""
        try:
            if category in self.categorized_memory:
                memories = self.categorized_memory[category]
                recent_memories = sorted(
                    memories.items(),
                    key=lambda x: x[1]['timestamp'],
                    reverse=True
                )[:5]
                
                return "\n".join([
                    f"- {memory['value']}"
                    for _, memory in recent_memories
                ])
            return ""
            
        except Exception as e:
            logging.error(f"Error getting category context: {e}")
            return ""

    def _calculate_word_distance(self, words: List[str], word1: str, word2: str) -> int:
        """Calculate the minimum distance between two words in a text"""
        positions1 = [i for i, word in enumerate(words) if word == word1]
        positions2 = [i for i, word in enumerate(words) if word == word2]
        
        if not positions1 or not positions2:
            return float('inf')
            
        return min(abs(p1 - p2) for p1 in positions1 for p2 in positions2)

    def _calculate_history_bonus(self, category: str, text: str) -> float:
        """Calculate bonus score based on conversation history"""
        recent_interactions = self.short_term_memory[-5:]  # Last 5 interactions
        category_count = sum(1 for interaction in recent_interactions 
                           if interaction.get('category') == category)
        
        # Small bonus for category continuity
        continuity_bonus = category_count * 0.2
        
        # Check for similar content in recent interactions
        similarity_bonus = 0.0
        for interaction in recent_interactions:
            if self._calculate_text_similarity(interaction['user_input'], text) > 0.5:
                similarity_bonus += 0.3
                
        return continuity_bonus + similarity_bonus

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple similarity between two texts"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0

    def add_interaction(self, user_input: str, response: str, category: Optional[MemoryCategory] = None):
        # Calculate importance
        context = self.get_recent_context(limit=3)
        importance = self._calculate_importance(user_input + response, context)
        
        # Create memory entry
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'response': response,
            'category': category.value if category else MemoryCategory.CONTEXTUAL.value,
            'importance': importance
        }
        
        # Add to short-term memory
        self.short_term_memory.append(interaction)
        
        # Generate and store embedding
        memory_key = f"mem_{len(self.memory_embeddings)}"
        self.memory_embeddings[memory_key] = self._embed_text(user_input + response)
        
        # If important enough, add to categorized memory
        if importance > self.importance_threshold:
            self.add_categorized_fact(
                category or MemoryCategory.CONTEXTUAL,
                f"interaction_{datetime.now().timestamp()}",
                f"{user_input} | {response}",
                {'importance': importance}
            )
        
        # Maintain memory limits
        self._maintain_memory_limits()
        self._save_memory()

    def _maintain_memory_limits(self, short_term_limit: int = 10):
        """Maintain memory limits while keeping most important memories"""
        if len(self.short_term_memory) > short_term_limit:
            # Sort by importance and recency
            sorted_memories = sorted(
                self.short_term_memory,
                key=lambda x: (x.get('importance', 0), x['timestamp']),
                reverse=True
            )
            self.short_term_memory = sorted_memories[:short_term_limit]

    def add_categorized_fact(self, category: MemoryCategory, key: str, value: str, metadata: Dict = None):
        if metadata is None:
            metadata = {}
            
        self.categorized_memory[category.value][key] = {
            'value': value,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata,
            'access_count': 0,
            'embedding_key': f"fact_{key}"
        }
        
        # Store embedding for the fact
        self.memory_embeddings[f"fact_{key}"] = self._embed_text(value)
        self._save_memory()

    def get_relevant_context(self, query: str, limit: int = 5, category: Optional[MemoryCategory] = None) -> str:
        """Get relevant context using semantic search"""
        query_embedding = self._embed_text(query)
        
        # Calculate similarities with all memories
        similarities = []
        
        # Check short-term memories
        for memory in self.short_term_memory:
            if category and memory['category'] != category.value:
                continue
            memory_text = f"{memory['user_input']} {memory['response']}"
            memory_embedding = self._embed_text(memory_text)
            similarity = cosine_similarity([query_embedding], [memory_embedding])[0][0]
            similarities.append((similarity, memory_text))
        
        # Check categorized memories
        for cat_memories in self.categorized_memory.values():
            for key, memory in cat_memories.items():
                if category and memory.get('category') != category.value:
                    continue
                embedding_key = memory.get('embedding_key')
                if embedding_key in self.memory_embeddings:
                    similarity = cosine_similarity(
                        [query_embedding],
                        [self.memory_embeddings[embedding_key]]
                    )[0][0]
                    similarities.append((similarity, memory['value']))
        
        # Sort by relevance and format
        similarities.sort(reverse=True)
        relevant_memories = similarities[:limit]
        
        return "\n".join(memory for _, memory in relevant_memories)

    def search_memory(self, query: str, category: Optional[MemoryCategory] = None) -> List[Dict]:
        """Enhanced semantic search through memories"""
        query_embedding = self._embed_text(query)
        results = []
        
        memories = (self.categorized_memory[category.value] 
                   if category 
                   else {k: v for d in self.categorized_memory.values() for k, v in d.items()})
        
        for key, memory in memories.items():
            embedding_key = memory.get('embedding_key')
            if embedding_key in self.memory_embeddings:
                similarity = cosine_similarity(
                    [query_embedding],
                    [self.memory_embeddings[embedding_key]]
                )[0][0]
                if similarity > 0.5:  # Similarity threshold
                    results.append({
                        'key': key,
                        'memory': memory,
                        'relevance': float(similarity)
                    })
        
        # Sort by relevance
        results.sort(key=lambda x: x['relevance'], reverse=True)
        return results