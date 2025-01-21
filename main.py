# main.py
import asyncio
import logging
import signal
from src.interfaces.telegram_interface import TelegramInterface
from src.utils.helpers import load_environment
from src.ai.web_learning.learning_service import LearningService
from src.ai.llm_handler import LLMHandler


def chunks(lst, n):
    """Helper function to split list into chunks"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class ApplicationManager:
    def __init__(self):
        self.env = load_environment()
        self.is_running = False
        self.tasks = set()
        self._cleanup_lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()
        self._initialized = False
        
    async def initialize_services(self):
        """Initialize all services in correct order"""
        if self._initialized:
            return True
            
        try:
            # Initialize LLM Handler
            self.llm_handler = LLMHandler()
            if not await self.llm_handler.initialize():
                raise RuntimeError("Failed to initialize LLM Handler")
            
            # Initialize Learning Service
            self.learning_service = LearningService()
            await self.learning_service.start()
            
            # Initialize Telegram Bot
            self.telegram_bot = TelegramInterface(
                token=self.env['TELEGRAM_TOKEN'],
                learning_service=self.learning_service
            )
            await self.telegram_bot.initialize()  # New initialize method
            
            self._initialized = True
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize services: {e}")
            await self._cleanup()
            return False

    async def start(self):
        """Start all services"""
        try:
            if not self._initialized and not await self.initialize_services():
                raise RuntimeError("Service initialization failed")
                
            self.is_running = True
            
            # Start telegram bot polling
            await self.telegram_bot.start()
            
            # Add monitoring tasks
            self.tasks.add(asyncio.create_task(
                self._monitor_health(),
                name="health_monitor"
            ))
            self.tasks.add(asyncio.create_task(
                self._monitor_telegram_connection(),
                name="telegram_monitor"
            ))
            
            logging.info("All services started successfully")
            
            await self._shutdown_event.wait()
            
        except Exception as e:
            logging.error(f"Error in application startup: {e}")
            self.is_running = False
        finally:
            await self._cleanup()

    async def _monitor_health(self):
        """Monitor health of services"""
        try:
            while not self._shutdown_event.is_set() and self.is_running:
                await asyncio.sleep(30)
                
                if not self.learning_service.is_healthy:
                    await self.shutdown("Learning service health check failed")
                    break
                    
                if not self.telegram_bot.is_healthy:
                    await self.shutdown("Telegram bot health check failed")
                    break
                    
        except asyncio.CancelledError:
            logging.info("Health monitor cancelled")
        except Exception as e:
            logging.error(f"Health monitoring error: {e}")
            await self.shutdown(f"Health monitoring failed: {e}")

    async def _monitor_telegram_connection(self):
        """Monitor telegram connection status"""
        retry_count = 0
        max_retries = 3
        
        try:
            while not self._shutdown_event.is_set() and self.is_running:
                await asyncio.sleep(60)
                
                if not self.telegram_bot.is_running:
                    retry_count += 1
                    logging.warning(f"Telegram connection lost. Retry {retry_count}/{max_retries}")
                    
                    if retry_count >= max_retries:
                        await self.shutdown("Max telegram reconnection attempts reached")
                        break
                        
                    try:
                        await self.telegram_bot.restart()
                        retry_count = 0
                    except Exception as e:
                        logging.error(f"Telegram restart failed: {e}")
                        
        except asyncio.CancelledError:
            logging.info("Telegram monitor cancelled")
        except Exception as e:
            logging.error(f"Telegram monitoring error: {e}")
            await self.shutdown(f"Telegram monitoring failed: {e}")
            

    async def _cleanup(self):
        """Clean up resources with proper locking"""
        async with self._cleanup_lock:
            try:
                # Set shutdown flags
                self._shutdown_event.set()
                self.is_running = False
                
                # First, handle running tasks with protection against recursion
                task_chunks = list(chunks(list(self.tasks), 10))  # Process tasks in chunks
                for chunk in task_chunks:
                    active_tasks = [t for t in chunk if not t.done()]
                    if active_tasks:
                        for task in active_tasks:
                            try:
                                task.cancel()
                            except Exception as e:
                                logging.error(f"Error cancelling task: {e}")
                        
                        await asyncio.shield(
                            asyncio.gather(*active_tasks, return_exceptions=True)
                        )
                
                # Stop services in reverse order with timeouts
                if hasattr(self, 'telegram_bot') and self.telegram_bot:
                    try:
                        await asyncio.wait_for(
                            self.telegram_bot.stop(),
                            timeout=10.0
                        )
                    except asyncio.TimeoutError:
                        logging.warning("Telegram bot shutdown timed out")
                    except Exception as e:
                        logging.error(f"Error stopping telegram bot: {e}")
                
                if hasattr(self, 'learning_service') and self.learning_service:
                    try:
                        await asyncio.wait_for(
                            self.learning_service.stop(),
                            timeout=10.0
                        )
                    except asyncio.TimeoutError:
                        logging.warning("Learning service shutdown timed out")
                    except Exception as e:
                        logging.error(f"Error stopping learning service: {e}")
                
                if hasattr(self, 'llm_handler') and self.llm_handler:
                    try:
                        await asyncio.wait_for(
                            self.llm_handler.cleanup(),
                            timeout=5.0
                        )
                    except asyncio.TimeoutError:
                        logging.warning("LLM handler cleanup timed out")
                    except Exception as e:
                        logging.error(f"Error cleaning up LLM handler: {e}")
                        
            except asyncio.CancelledError:
                logging.info("Cleanup cancelled, completing critical operations")
            except Exception as e:
                logging.error(f"Error during cleanup: {e}")
            finally:
                try:
                    # Final cleanup of any remaining tasks
                    remaining_tasks = [t for t in asyncio.all_tasks() 
                                    if t is not asyncio.current_task() 
                                    and not t.done()]
                    if remaining_tasks:
                        for task in remaining_tasks:
                            task.cancel()
                        await asyncio.gather(*remaining_tasks, return_exceptions=True)
                    
                    self.tasks.clear()
                    self._initialized = False
                    logging.info("Cleanup completed successfully")
                except Exception as e:
                    logging.error(f"Error during final cleanup: {e}")

    

    async def shutdown(self, reason: str = "Shutdown requested"):
        """Graceful shutdown"""
        if not self._shutdown_event.is_set():
            logging.info(f"Shutting down. Reason: {reason}")
            self._shutdown_event.set()
            self.is_running = False

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('application.log')
        ]
    )

async def async_main():
    """Async main function with proper shutdown handling"""
    app = ApplicationManager()
    
    def signal_handler(sig):
        asyncio.create_task(app.shutdown(f"Signal {sig.name} received"))
    
    # Setup signal handlers
    for sig in (signal.SIGTERM, signal.SIGINT):
        asyncio.get_running_loop().add_signal_handler(
            sig, 
            lambda s=sig: signal_handler(s)
        )
    
    try:
        await app.start()
    except asyncio.CancelledError:
        pass
    finally:
        await app.shutdown("Main loop ended")
        await app._cleanup()

def main():
    """Main entry point with proper exception handling"""
    setup_logging()
    
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        logging.info("Application stopped by user")
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)

if __name__ == '__main__':
    main()