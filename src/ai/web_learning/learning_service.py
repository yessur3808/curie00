# src/ai/web_learning/learning_service.py

import asyncio
from typing import Optional, Dict, Any
import logging
from datetime import datetime
import signal
from collections import deque
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import newspaper
from scholarly import scholarly
from .web_scraper import EnhancedWebLearner
from .config.domain_config import TRUSTED_DOMAINS, DOMAIN_WEIGHTS

class LearningService:
    def __init__(self):
        self.learner = EnhancedWebLearner()
        self.learning_queue = asyncio.Queue()
        self.learning_tasks = set()
        self.is_running = False
        self.learning_history = deque(maxlen=1000)
        self.current_topics = set()
        self.session = None
        self._shutdown_event = asyncio.Event()
        self._cleanup_timeout = 10
        self._health_check_interval = 30
        self._last_health_check = datetime.now()
        self._health_status = True
        
        
    @property
    def is_healthy(self) -> bool:
        """Check if service is healthy"""
        return (
            self.is_running and 
            self._health_status and 
            not self._shutdown_event.is_set() and
            hasattr(self, 'learner') and 
            self.learner is not None
        )
        
    @property
    def active_topics(self) -> set:
        """Get currently active learning topics"""
        return self.current_topics

    @property
    def max_parallel(self) -> int:
        """Maximum number of parallel learning tasks"""
        return 5  # Adjust as needed

    async def get_topic_summary(self, topic: str) -> Optional[Dict[str, Any]]:
        """Get summary of learned topic"""
        try:
            # Check learning history
            for entry in reversed(self.learning_history):
                if entry['topic'].lower() == topic.lower():
                    knowledge = entry['result'].get('enhanced_knowledge', {})
                    return {
                        'key_points': knowledge.get('key_points', []),
                        'confidence': knowledge.get('confidence', 0.0),
                        'sources': knowledge.get('sources', [])
                    }
            return None
        except Exception as e:
            logging.error(f"Error getting topic summary: {e}")
            return None

    async def _health_check(self) -> bool:
        """Perform health check"""
        try:
            # Check core components
            if not self.learner:
                return False
                
            # Check queue processor
            if hasattr(self, '_queue_processor'):
                if self._queue_processor.done():
                    exception = self._queue_processor.exception()
                    if exception:
                        logging.error(f"Queue processor failed: {exception}")
                        return False
                        
            # Check session
            if not self.session or self.session.closed:
                return False
                
            self._last_health_check = datetime.now()
            return True
            
        except Exception as e:
            logging.error(f"Health check failed: {e}")
            return False

    async def _extract_knowledge(self, sources: list, topic: str) -> Dict[str, Any]:
        """Extract enhanced knowledge from sources"""
        try:
            key_points = []
            sources_processed = []
            confidence = 0.0
            
            for source in sources:
                try:
                    if isinstance(source, dict):
                        url = source.get('url', '')
                    else:
                        url = source
                        
                    if not url:
                        continue
                        
                    article = newspaper.Article(url)
                    await self._async_download(article)
                    article.parse()
                    article.nlp()
                    
                    key_points.extend(article.keywords[:5])
                    sources_processed.append({
                        'url': url,
                        'title': article.title,
                        'summary': article.summary
                    })
                    confidence += source.get('score', 0.5) if isinstance(source, dict) else 0.5
                    
                except Exception as e:
                    logging.warning(f"Error processing source {url}: {e}")
                    continue
                    
            return {
                'key_points': list(set(key_points)),
                'sources': sources_processed,
                'confidence': min(confidence / len(sources) if sources else 0.0, 1.0)
            }
            
        except Exception as e:
            logging.error(f"Knowledge extraction error: {e}")
            return {
                'key_points': [],
                'sources': [],
                'confidence': 0.0
            }

    async def _async_download(self, article):
        """Asynchronously download article"""
        if self.session and not self.session.closed:
            async with self.session.get(article.url) as response:
                article.html = await response.text()
                article.download_state = 2  # Mark as downloaded
        
    async def start(self):
        """Start the learning service"""
        try:
            self.is_running = True
            self.session = aiohttp.ClientSession()
            self._queue_processor = asyncio.create_task(self._process_queue())
            logging.info("Learning service started successfully")
        except Exception as e:
            logging.error(f"Failed to start learning service: {e}")
            await self.stop()
            raise
        
    async def stop(self):
        """Stop the learning service gracefully"""
        try:
            self._shutdown_event.set()
            self.is_running = False
            
            # Handle queue processor with proper task cancellation
            if hasattr(self, '_queue_processor'):
                try:
                    self._queue_processor.cancel()
                    try:
                        # Wait for cancellation to complete
                        await self._queue_processor
                    except asyncio.CancelledError:
                        logging.info("Queue processor cancelled successfully")
                    except Exception as e:
                        logging.error(f"Error during queue processor cancellation: {e}")
                except Exception as e:
                    logging.error(f"Failed to cancel queue processor: {e}")

            # Handle learning tasks with shield
            if self.learning_tasks:
                try:
                    # Cancel all tasks
                    for task in self.learning_tasks:
                        if not task.done():
                            task.cancel()
                    
                    # Wait for all tasks with shield to prevent premature cancellation
                    await asyncio.shield(
                        asyncio.gather(*self.learning_tasks, return_exceptions=True)
                    )
                except asyncio.CancelledError:
                    logging.info("Learning tasks cancelled successfully")
                except Exception as e:
                    logging.error(f"Error during task cleanup: {e}")

            # Handle session cleanup
            if self.session and not self.session.closed:
                try:
                    await asyncio.shield(self.session.close())
                except Exception as e:
                    logging.error(f"Error closing session: {e}")

            # Handle learner cleanup with shield
            try:
                await asyncio.shield(
                    asyncio.wait_for(
                        self.learner.cleanup(),
                        timeout=self._cleanup_timeout
                    )
                )
            except asyncio.TimeoutError:
                logging.warning("Learner cleanup timed out")
            except asyncio.CancelledError:
                logging.info("Learner cleanup cancelled")
            except Exception as e:
                logging.error(f"Error during learner cleanup: {e}")

        except asyncio.CancelledError:
            logging.info("Stop operation cancelled, completing critical cleanup")
        except Exception as e:
            logging.error(f"Error during service shutdown: {e}")
        finally:
            # Final cleanup in case anything was missed
            try:
                remaining_tasks = [
                    task for task in asyncio.all_tasks() 
                    if task is not asyncio.current_task()
                    and not task.done()
                    and not task.cancelled()
                ]
                
                if remaining_tasks:
                    for task in remaining_tasks:
                        task.cancel()
                    await asyncio.gather(*remaining_tasks, return_exceptions=True)
                    
            except Exception as e:
                logging.error(f"Error during final cleanup: {e}")
            finally:
                logging.info("Learning service stopped gracefully")

    async def _process_queue(self):
        """Process the learning queue"""
        try:
            while not self._shutdown_event.is_set():
                try:
                    if not self.learning_queue.empty():
                        topic, sources = await self.learning_queue.get()
                        if not self._shutdown_event.is_set():  # Double check before creating new task
                            task = asyncio.create_task(self._learn_topic(topic, sources))
                            self.learning_tasks.add(task)
                            task.add_done_callback(self.learning_tasks.discard)
                    await asyncio.sleep(1)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logging.error(f"Error processing learning queue: {e}")
                    await asyncio.sleep(1)  # Prevent tight loop on continuous errors
        except asyncio.CancelledError:
            logging.info("Queue processor shutdown initiated")
        finally:
            self.is_running = False

    async def queue_topic(self, topic: str, sources: Optional[list] = None) -> bool:
        """Queue a topic for learning"""
        try:
            if self._shutdown_event.is_set():
                return False
                
            if topic not in self.current_topics:
                if not sources:
                    sources = await self._discover_sources(topic)
                await self.learning_queue.put((topic, sources))
                self.current_topics.add(topic)
                logging.info(f"Topic queued for learning: {topic}")
                return True
            return False
        except Exception as e:
            logging.error(f"Error queueing topic: {e}")
            return False
        


    async def _learn_topic(self, topic: str, sources: list):
        """Process a single topic with enhanced learning"""
        try:
            # Validate and score sources
            validated_sources = await self._validate_sources(sources)
            
            # Use existing learner for base knowledge
            base_result = await self.learner.learn_topic(topic, validated_sources)
            
            # Extract additional knowledge
            enhanced_knowledge = await self._extract_knowledge(validated_sources, topic)
            
            # Combine results
            result = {
                'base_knowledge': base_result,
                'enhanced_knowledge': enhanced_knowledge,
                'sources': validated_sources
            }
            
            self.learning_history.append({
                'topic': topic,
                'timestamp': datetime.now().isoformat(),
                'result': result
            })
            
            self.current_topics.remove(topic)
            logging.info(f"Topic learned successfully: {topic}")
            return result
            
        except Exception as e:
            logging.error(f"Error learning topic {topic}: {e}")
            self.current_topics.remove(topic)
            return {'error': str(e)}

    async def _discover_sources(self, topic: str):
        """Discover legitimate sources for learning"""
        sources = set()
        
        # Academic papers
        try:
            search_query = scholarly.search_pubs(topic)
            papers = [next(search_query) for _ in range(5)]
            sources.update([paper.get('url') for paper in papers if paper.get('url')])
        except Exception as e:
            logging.warning(f"Scholar search failed: {e}")

        # Technical documentation
        if any(tech_term in topic.lower() for tech_term in ['programming', 'development', 'code', 'software']):
            sources.update(await self._find_technical_docs(topic))

        # Stack Overflow and other sources
        if self.session:
            try:
                async with self.session.get(
                    f"https://api.stackexchange.com/2.3/search?order=desc&sort=votes&intitle={topic}&site=stackoverflow"
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        sources.update([item['link'] for item in data.get('items', [])])
            except Exception as e:
                logging.warning(f"Stack Overflow API error: {e}")

        return list(sources)

    async def _validate_sources(self, sources: list):
        """Validate and rank sources"""
        validated = []
        
        for url in sources:
            try:
                score = await self._score_source(url)
                if score > 0.6:  # Acceptance threshold
                    validated.append({
                        'url': url,
                        'score': score
                    })
            except Exception as e:
                logging.warning(f"Source validation error for {url}: {e}")
                
        return sorted(validated, key=lambda x: x['score'], reverse=True)

    async def _score_source(self, url: str) -> float:
        """Score a source based on various factors"""
        if not url:
            return 0.0
            
        score = 0.0
        domain = urlparse(url).netloc
        
        # Domain reputation scoring
        if any(domain.endswith(tld) for tld in self.trusted_domains['educational']):
            score += 0.4
        elif any(domain in tld for tld in self.trusted_domains['scientific']):
            score += 0.35
        elif any(domain in tld for tld in self.trusted_domains['documentation']):
            score += 0.3
        elif any(domain in tld for tld in self.trusted_domains['community']):
            score += 0.25

        return min(score + 0.3, 1.0)  # Base score + reputation score, capped at 1.0