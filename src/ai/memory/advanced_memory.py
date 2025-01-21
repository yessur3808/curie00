# src/ai/memory/advanced_memory.py
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class MemoryEmbedding:
    content: str
    embedding: np.ndarray
    timestamp: datetime
    metadata: Dict[str, Any]

class AdvancedMemorySystem:
    def __init__(self, memory_path: str = "data/memory"):
        self.memory_path = Path(memory_path)
        self.memory_path.mkdir(parents=True, exist_ok=True)
        
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.episodic_memory: List[Dict] = []
        self.semantic_memory: Dict[str, MemoryEmbedding] = {}
        self.working_memory: List[Dict] = []
        self.importance_threshold = 0.7
        self.lock = asyncio.Lock()  # Add lock for thread safety
        
        self._load_memories()

    def _embed_text(self, text: str) -> np.ndarray:
        """Synchronous embedding generation"""
        return self.embedding_model.encode([text])[0]

    async def _async_embed_text(self, text: str) -> np.ndarray:
        """Asynchronous wrapper for embedding generation"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._embed_text, text)

    async def _calculate_importance(self, text: str, context: str) -> float:
        """Asynchronously calculate importance score"""
        text_embedding = await self._async_embed_text(text)
        context_embedding = await self._async_embed_text(context)
        return float(cosine_similarity([text_embedding], [context_embedding])[0][0])

    async def add_interaction(self, user_input: str, response: str, context: str = "") -> None:
        """Asynchronously add new interaction"""
        async with self.lock:
            importance = await self._calculate_importance(user_input + response, context)
            
            memory_entry = {
                'timestamp': datetime.now().isoformat(),
                'user_input': user_input,
                'response': response,
                'importance': importance,
                'embedding': (await self._async_embed_text(user_input + response)).tolist(),
                'context': context
            }
            
            self.episodic_memory.append(memory_entry)
            
            if importance > self.importance_threshold:
                await self._add_to_semantic_memory(user_input, response, context)
            
            await self._maintain_memory_limits()
            await self._save_memories()

    async def _add_to_semantic_memory(self, input_text: str, response: str, context: str) -> None:
        """Asynchronously add to semantic memory"""
        async with self.lock:
            combined_text = f"{input_text} | {response}"
            embedding = await self._async_embed_text(combined_text)
            
            memory = MemoryEmbedding(
                content=combined_text,
                embedding=embedding,
                timestamp=datetime.now(),
                metadata={'context': context}
            )
            
            key = f"memory_{len(self.semantic_memory)}"
            self.semantic_memory[key] = memory

    async def get_relevant_context(self, query: str, limit: int = 5) -> List[Dict]:
        """Asynchronously retrieve relevant memories"""
        query_embedding = await self._async_embed_text(query)
        all_memories = []
        
        async with self.lock:
            # Process episodic memories
            for memory in self.episodic_memory:
                embedding = np.array(memory['embedding'])
                similarity = cosine_similarity([query_embedding], [embedding])[0][0]
                all_memories.append((similarity, {
                    'type': 'episodic',
                    'content': f"User: {memory['user_input']}\nAssistant: {memory['response']}",
                    'timestamp': memory['timestamp']
                }))
            
            # Process semantic memories
            for key, memory in self.semantic_memory.items():
                similarity = cosine_similarity([query_embedding], [memory.embedding])[0][0]
                all_memories.append((similarity, {
                    'type': 'semantic',
                    'content': memory.content,
                    'timestamp': memory.timestamp.isoformat()
                }))
        
        all_memories.sort(key=lambda x: x[0], reverse=True)
        return [mem[1] for mem in all_memories[:limit]]

    async def _maintain_memory_limits(self):
        """Asynchronously maintain memory limits"""
        async with self.lock:
            if len(self.episodic_memory) > 100:
                self.episodic_memory.sort(
                    key=lambda x: (x['importance'], x['timestamp']), 
                    reverse=True
                )
                self.episodic_memory = self.episodic_memory[:100]
            
            if len(self.semantic_memory) > 1000:
                sorted_memories = sorted(
                    self.semantic_memory.items(),
                    key=lambda x: x[1].timestamp
                )
                self.semantic_memory = dict(sorted_memories[-1000:])

    async def _save_memories(self):
        """Asynchronously save memories"""
        async with self.lock:
            memory_data = {
                'episodic': self.episodic_memory,
                'semantic': {
                    k: {
                        'content': v.content,
                        'embedding': v.embedding.tolist(),
                        'timestamp': v.timestamp.isoformat(),
                        'metadata': v.metadata
                    } for k, v in self.semantic_memory.items()
                }
            }
            
            with open(self.memory_path / 'memories.json', 'w') as f:
                json.dump(memory_data, f, indent=2)

    def _load_memories(self):
        """Load memories (sync because called in __init__)"""
        try:
            with open(self.memory_path / 'memories.json', 'r') as f:
                data = json.load(f)
                
                self.episodic_memory = data.get('episodic', [])
                
                for k, v in data.get('semantic', {}).items():
                    self.semantic_memory[k] = MemoryEmbedding(
                        content=v['content'],
                        embedding=np.array(v['embedding']),
                        timestamp=datetime.fromisoformat(v['timestamp']),
                        metadata=v['metadata']
                    )
        except FileNotFoundError:
            pass