# src/ai/web_learning/web_scraper.py

import requests
from bs4 import BeautifulSoup
import asyncio
import aiohttp
import logging
from urllib.parse import urljoin, urlparse
from datetime import datetime
import json
from newspaper import Article
import trafilatura
from concurrent.futures import ThreadPoolExecutor
import re
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dataclasses import dataclass
from enum import Enum
import hashlib
from typing import List, Dict, Set, Optional, Tuple, Any, TypeVar, Union, Callable, Iterable
from collections.abc import Iterable

class SourceTier(Enum):
    ACADEMIC = 4  # .edu, academic journals
    VERIFIED = 3  # Verified educational/scientific sites
    RELIABLE = 2  # Well-known reliable sources
    GENERAL = 1   # General web content
    UNTRUSTED = 0 # Unreliable or unverified sources

@dataclass
class VerificationResult:
    is_valid: bool
    confidence: float
    source_tier: SourceTier
    cross_references: int
    contradictions: List[str]

class ContentValidator:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.quality_threshold = 0.75
        self.min_topic_relevance = 0.6
        
    def validate_content(self, content: str, topic: str) -> Tuple[bool, float, Dict]:
        doc = self.nlp(content)
        topic_doc = self.nlp(topic)
        
        # Calculate semantic similarity with topic
        topic_similarity = doc.similarity(topic_doc)
        
        # Calculate content quality metrics
        metrics = {
            'coherence': self._measure_coherence(doc),
            'information_density': self._calculate_info_density(doc),
            'technical_accuracy': self._assess_technical_accuracy(doc),
            'citation_presence': self._check_citations(content),
            'topic_relevance': topic_similarity
        }
        
        # Calculate overall quality score
        quality_score = sum(metrics.values()) / len(metrics)
        is_valid = quality_score > self.quality_threshold and topic_similarity > self.min_topic_relevance
        
        return is_valid, quality_score, metrics
    
    def _measure_coherence(self, doc) -> float:
        # Measure text coherence using linguistic features
        return min(1.0, len([sent for sent in doc.sents]) / 50)
    
    def _calculate_info_density(self, doc) -> float:
        # Calculate information density using NER and key phrase density
        entities = len(doc.ents)
        words = len(doc)
        return min(1.0, entities / (words + 1) * 10)
    
    def _assess_technical_accuracy(self, doc) -> float:
        # Assess technical accuracy using domain-specific terminology
        technical_terms = len([token for token in doc if token.pos_ in ['NOUN', 'PROPN']])
        return min(1.0, technical_terms / len(doc) * 5)
    
    def _check_citations(self, content: str) -> float:
        # Check for presence of citations and references
        citation_patterns = [r'\[\d+\]', r'\(\d{4}\)', r'et al\.']
        citations = sum(len(re.findall(pattern, content)) for pattern in citation_patterns)
        return min(1.0, citations / 10)

class SourceValidator:
    def __init__(self):
        self.academic_domains = {'.edu', '.ac.uk', '.ac.jp'}
        self.verified_domains = set()  # Load from configuration
        self.blocked_domains = set()   # Load from configuration
        
    def validate_source(self, url: str) -> Tuple[SourceTier, float]:
        domain = urlparse(url).netloc
        
        # Check for academic domains
        if any(domain.endswith(ac_dom) for ac_dom in self.academic_domains):
            return SourceTier.ACADEMIC, 0.95
            
        # Check for verified domains
        if domain in self.verified_domains:
            return SourceTier.VERIFIED, 0.85
            
        # Check for blocked domains
        if domain in self.blocked_domains:
            return SourceTier.UNTRUSTED, 0.0
            
        # Default to general tier with moderate confidence
        return SourceTier.GENERAL, 0.6

class FactChecker:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.fact_embeddings = {}
        self.contradiction_threshold = 0.85
        
    async def verify_fact(self, fact: str, sources: List[str]) -> VerificationResult:
        fact_doc = self.nlp(fact)
        fact_hash = self._hash_fact(fact)
        
        confirmations = 0
        contradictions = []
        
        for source in sources:
            similarity = await self._compare_with_source(fact_doc, source)
            if similarity > self.contradiction_threshold:
                confirmations += 1
            elif similarity < -self.contradiction_threshold:
                contradictions.append(source)
                
        confidence = self._calculate_confidence(confirmations, len(contradictions))
        
        return VerificationResult(
            is_valid=confidence > 0.7,
            confidence=confidence,
            source_tier=self._determine_source_tier(sources),
            cross_references=confirmations,
            contradictions=contradictions
        )
        
    def _hash_fact(self, fact: str) -> str:
        return hashlib.sha256(fact.encode()).hexdigest()
    
    async def _compare_with_source(self, fact_doc, source: str) -> float:
        # Implement source comparison logic
        return 0.8  # Placeholder
    
    def _calculate_confidence(self, confirmations: int, contradictions: int) -> float:
        if contradictions > confirmations:
            return 0.0
        return min(1.0, confirmations / (confirmations + contradictions + 1))
    
    def _determine_source_tier(self, sources: List[str]) -> SourceTier:
        # Implement source tier determination logic
        return SourceTier.GENERAL


class WebLearner:
    """Base class for web learning functionality"""
    def __init__(self, max_depth: int = 2, max_pages: int = 10):
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.nlp = spacy.load('en_core_web_sm')
        self.visited_urls: Set[str] = set()
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def initialize(self):
        """Initialize async resources"""
        self.session = aiohttp.ClientSession()
        
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
            
    async def fetch_content(self, url: str) -> Optional[str]:
        """Fetch content from URL using multiple methods"""
        try:
            # Try trafilatura first
            downloaded = await self._async_fetch(url)
            if downloaded:
                content = trafilatura.extract(downloaded)
                if content:
                    return content
                    
            # Fallback to newspaper3k
            article = Article(url)
            await asyncio.get_event_loop().run_in_executor(None, article.download)
            await asyncio.get_event_loop().run_in_executor(None, article.parse)
            if article.text:
                return article.text
                
            # Last resort: basic HTML parsing
            if self.session:
                async with self.session.get(url) as response:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'lxml')
                    return ' '.join([p.text for p in soup.find_all('p')])
                    
            return None
            
        except Exception as e:
            logging.error(f"Error fetching content from {url}: {str(e)}")
            return None
            
    async def _async_fetch(self, url: str) -> Optional[str]:
        """Async fetch using trafilatura"""
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None, trafilatura.fetch_url, url
            )
        except Exception:
            return None
            
    async def extract_links(self, url: str, html: str) -> List[str]:
        """Extract valid links from HTML content"""
        try:
            soup = BeautifulSoup(html, 'lxml')
            links = []
            for link in soup.find_all('a', href=True):
                absolute_url = urljoin(url, link['href'])
                if self._is_valid_url(absolute_url):
                    links.append(absolute_url)
            return links[:self.max_pages]
        except Exception as e:
            logging.error(f"Error extracting links from {url}: {str(e)}")
            return []
            
    def _is_valid_url(self, url: str) -> bool:
        """Validate URL"""
        try:
            parsed = urlparse(url)
            return all([parsed.scheme, parsed.netloc])
        except:
            return False

class EnhancedWebLearner(WebLearner):
    def __init__(self, max_depth: int = 2, max_pages: int = 10):
        super().__init__(max_depth, max_pages)
        self.content_validator = ContentValidator()
        self.source_validator = SourceValidator()
        self.fact_checker = FactChecker()
        self.verified_knowledge: Dict[str, List[Dict]] = {}
        self._knowledge_cache: Dict[str, Dict] = {}
        self._last_cleanup = datetime.now()

    async def get_topic_summary(self, topic: str) -> Optional[Dict[str, Any]]:
        """Get summary of learned topic with caching"""
        try:
            # Check cache first
            if topic in self._knowledge_cache:
                cached = self._knowledge_cache[topic]
                if (datetime.now() - cached['timestamp']).total_seconds() < 3600:  # 1 hour cache
                    return cached['data']

            # Check verified knowledge
            if topic in self.verified_knowledge:
                knowledge = self.verified_knowledge[topic]
                summary = {
                    'key_points': [fact['content'] for fact in knowledge[:5]],
                    'confidence': self._calculate_overall_confidence(knowledge),
                    'sources': [fact.get('source', '') for fact in knowledge],
                    'timestamp': datetime.now().isoformat()
                }
                
                # Cache the result
                self._knowledge_cache[topic] = {
                    'data': summary,
                    'timestamp': datetime.now()
                }
                
                return summary
            return None
            
        except Exception as e:
            logging.error(f"Error getting topic summary: {e}")
            return None

    async def cleanup(self):
        """Cleanup resources and cache"""
        try:
            # Clear caches
            self._knowledge_cache.clear()
            
            # Cleanup validators and checkers
            if hasattr(self.content_validator, 'cleanup'):
                await self.content_validator.cleanup()
            if hasattr(self.source_validator, 'cleanup'):
                await self.source_validator.cleanup()
            if hasattr(self.fact_checker, 'cleanup'):
                await self.fact_checker.cleanup()
            
            # Clear verified knowledge
            self.verified_knowledge.clear()
            
            logging.info("EnhancedWebLearner cleanup completed")
            
        except Exception as e:
            logging.error(f"Error during EnhancedWebLearner cleanup: {e}")

    async def learn_topic(self, topic: str, sources: List[str]) -> Dict:
        """Enhanced learning with verification and fact-checking"""
        try:
            # Check cache first
            cache_key = f"{topic}_{hash(tuple(sorted(sources)))}"
            if cache_key in self._knowledge_cache:
                cached = self._knowledge_cache[cache_key]
                if (datetime.now() - cached['timestamp']).total_seconds() < 3600:
                    return cached['data']

            # Your existing learn_topic implementation
            verified_sources = []
            for source in sources:
                tier, confidence = self.source_validator.validate_source(source)
                if tier in [SourceTier.ACADEMIC, SourceTier.VERIFIED, SourceTier.RELIABLE]:
                    verified_sources.append((source, tier, confidence))
                    
            if not verified_sources:
                return {'error': 'No reliable sources found'}

            results = []
            async with asyncio.TaskGroup() as group:
                for source, tier, confidence in verified_sources:
                    task = group.create_task(
                        self._enhanced_scrape_source(topic, source, tier, confidence)
                    )
                    results.append(task)

            verified_knowledge = []
            for result in results:
                if result.done():
                    knowledge = await result
                    if knowledge:
                        verified_facts = await self._verify_knowledge(
                            knowledge, 
                            verified_sources
                        )
                        verified_knowledge.extend(verified_facts)

            self.verified_knowledge[topic] = verified_knowledge
            
            result = {
                'topic': topic,
                'verified_sources': len(verified_sources),
                'verified_facts': len(verified_knowledge),
                'confidence_score': self._calculate_overall_confidence(verified_knowledge),
                'timestamp': datetime.now().isoformat(),
                'key_points': [fact['content'] for fact in verified_knowledge[:5]]
            }

            # Cache the result
            self._knowledge_cache[cache_key] = {
                'data': result,
                'timestamp': datetime.now()
            }

            return result

        except Exception as e:
            logging.error(f"Error in enhanced learning for topic {topic}: {str(e)}")
            return {'error': str(e)}

    @property
    def is_healthy(self) -> bool:
        """Check if learner is healthy"""
        return (
            hasattr(self, 'content_validator') and
            hasattr(self, 'source_validator') and
            hasattr(self, 'fact_checker') and
            all([
                self.content_validator is not None,
                self.source_validator is not None,
                self.fact_checker is not None
            ])
        )

    async def _cache_maintenance(self):
        """Perform cache maintenance"""
        try:
            current_time = datetime.now()
            expired_keys = [
                k for k, v in self._knowledge_cache.items()
                if (current_time - v['timestamp']).total_seconds() > 3600
            ]
            for k in expired_keys:
                del self._knowledge_cache[k]
        except Exception as e:
            logging.error(f"Cache maintenance error: {e}")