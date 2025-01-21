from telegram.ext import Application, CommandHandler, MessageHandler, filters
from telegram import Update
from src.bot.core import CurieBot
from src.ai.llm_handler import LLMHandler
from src.ai.web_learning.learning_service import LearningService
import logging
import asyncio
from typing import Optional, Dict
from datetime import datetime

class TelegramInterface:
    def __init__(self, token: str, learning_service: Optional[LearningService] = None):
        self.token = token
        self.bot = CurieBot()
        self.app = Application.builder().token(self.token).build()
        self.llm = LLMHandler()
        self.learning_service = learning_service or LearningService()
        self.learning_notifications: Dict[int, Dict] = {}
        self._initialized = False
        self._tasks = set()
        self._shutdown_event = asyncio.Event()
        self.is_running = False
        self.is_healthy = True  # Add health status
        self._polling_task = None

    async def initialize(self) -> bool:
        """Initialize the telegram interface"""
        if self._initialized:
            return True

        try:
            await self.app.initialize()
            await self.llm.initialize()
            self.setup_handlers()
            self._initialized = True
            return True

        except Exception as e:
            logging.error(f"Failed to initialize Telegram interface: {e}")
            self.is_healthy = False
            return False
        
    async def start(self):
        """Start the bot and learning service"""
        if self.is_running:
            return

        try:
            if not self._initialized and not await self.initialize():
                raise RuntimeError("Failed to initialize Telegram interface")

            await self.learning_service.start()
            await self.app.start()
            self.is_running = True
            
            # Start polling in a protected task
            self._polling_task = asyncio.create_task(
                self._protected_polling(),
                name="telegram_polling"
            )
            
            await self._shutdown_event.wait()
            
        except Exception as e:
            logging.error(f"Failed to start Telegram interface: {e}")
            self.is_healthy = False
            raise
            
        except Exception as e:
            logging.error(f"Failed to start Telegram interface: {e}")
            raise
        finally:
            await self.stop()

    async def _run_polling(self):
        """Run polling in a separate task"""
        try:
            await self.app.run_polling(drop_pending_updates=True)
        except asyncio.CancelledError:
            logging.info("Polling task cancelled")
        except Exception as e:
            logging.error(f"Polling error: {e}")
            await self.stop()

    async def _protected_polling(self):
        """Protected polling to prevent recursion"""
        try:
            await self.app.run_polling(drop_pending_updates=True, close_loop=False)
        except asyncio.CancelledError:
            logging.info("Polling task cancelled")
        except Exception as e:
            logging.error(f"Polling error: {e}")
            self.is_healthy = False
            self._shutdown_event.set()

    async def stop(self):
        """Stop the bot and learning service gracefully"""
        if not self.is_running:
            return

        self.is_running = False
        self._shutdown_event.set()

        try:
            # Stop application first
            if self.app:
                try:
                    # Use shield to prevent cancellation during cleanup
                    await asyncio.shield(
                        asyncio.wait_for(
                            self.app.stop(),
                            timeout=5.0
                        )
                    )
                except asyncio.TimeoutError:
                    logging.warning("Telegram app shutdown timed out")
                except Exception as e:
                    logging.error(f"Error stopping Telegram app: {e}")

            # Cancel polling task
            if self._polling_task and not self._polling_task.done():
                self._polling_task.cancel()
                try:
                    await self._polling_task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logging.error(f"Error cancelling polling task: {e}")

            # Stop learning service
            if self.learning_service:
                try:
                    await asyncio.shield(
                        asyncio.wait_for(
                            self.learning_service.stop(),
                            timeout=5.0
                        )
                    )
                except asyncio.TimeoutError:
                    logging.warning("Learning service shutdown timed out")
                except Exception as e:
                    logging.error(f"Error stopping learning service: {e}")

            self._initialized = False
            
        except Exception as e:
            logging.error(f"Error during Telegram interface shutdown: {e}")
            self.is_healthy = False

    def is_healthy(self) -> bool:
        """Check if the interface is healthy"""
        return (
            self.is_healthy and 
            self.is_running and 
            self._initialized and 
            (self._polling_task is None or not self._polling_task.done())
        )
        
        
    async def start_command(self, update: Update, context) -> None:
        welcome_message = """
        👋 Hello! I'm your AI-powered assistant. I can help you with:
        
        🤖 General questions and discussions
        📚 Information and explanations
        🎯 Problem-solving and advice
        🧠 Learning new topics (/learn command)
        
        Commands:
        /learn <topic> - I'll research and learn about a specific topic
        /status - Check the status of learning tasks
        /help - Show this help message
        
        Just send me a message and I'll do my best to help!
        """
        await update.message.reply_text(welcome_message)

    async def help_command(self, update: Update, context) -> None:
        help_message = """
        🤖 Available Commands:
        
        /learn <topic> - Start learning about a specific topic
        /status - Check status of learning tasks
        /help - Show this help message
        
        Examples:
        /learn quantum computing
        /learn artificial intelligence ethics
        """
        await update.message.reply_text(help_message)

    async def learn_command(self, update: Update, context) -> None:
        try:
            if not context.args:
                await update.message.reply_text(
                    "Please specify what you'd like me to learn about.\n"
                    "Usage: /learn <topic>\n"
                    "Example: /learn quantum computing"
                )
                return
                
            topic = ' '.join(context.args)
            user_id = update.effective_user.id
            
            # Queue the learning task
            await self.learning_service.queue_topic(topic)
            
            # Store the learning request
            self.learning_notifications[user_id] = {
                'topic': topic,
                'timestamp': datetime.now(),
                'status': 'queued'
            }
            
            await update.message.reply_text(
                f"🧠 I'm starting to learn about: {topic}\n"
                "This may take a few minutes. I'll notify you when I'm done!\n"
                "You can check the status with /status"
            )
            
            # Start monitoring the learning progress
            asyncio.create_task(self._monitor_learning_progress(update, topic))
            
        except Exception as e:
            logging.error(f"Error in learn command: {e}")
            await update.message.reply_text(
                "Sorry, I encountered an error while processing your learning request."
            )

    async def status_command(self, update: Update, context) -> None:
        user_id = update.effective_user.id
        
        if user_id in self.learning_notifications:
            learning_info = self.learning_notifications[user_id]
            status_message = (
                f"🧠 Learning Status for: {learning_info['topic']}\n"
                f"Status: {learning_info['status']}\n"
                f"Started: {learning_info['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"
            )
        else:
            status_message = "No active learning tasks found."
            
        await update.message.reply_text(status_message)

    async def handle_message(self, update: Update, context) -> None:
        user_message = update.message.text
        user_id = update.message.from_user.id
        
        # Let user know we're processing
        await update.message.reply_chat_action("typing")
        
        try:
            # Check if there's relevant learned knowledge
            learned_context = await self._get_relevant_learned_context(user_message)
            
            # Construct prompt with system context and learned knowledge
            prompt = (
                f"{self.llm.get_system_prompt()}\n\n"
                f"{learned_context}\n\n"
                f"User: {user_message}\nAssistant:"
            )
            
            # Get AI response
            response = self.llm.generate_response(prompt)
            
            # Send response
            await update.message.reply_text(response)
            
        except Exception as e:
            logging.error(f"Error processing message: {e}")
            await update.message.reply_text(
                "Sorry, I'm having trouble thinking right now. Please try again!"
            )

    async def _monitor_learning_progress(self, update: Update, topic: str):
        """Monitor the progress of a learning task"""
        user_id = update.effective_user.id
        try:
            while user_id in self.learning_notifications:
                # Check if topic has been learned
                if topic not in self.learning_service.current_topics:
                    learned_info = await self._get_topic_summary(topic)
                    if learned_info:
                        await update.message.reply_text(
                            f"✅ I've finished learning about {topic}!\n\n"
                            f"Key points learned:\n{learned_info['summary']}\n\n"
                            f"Confidence: {learned_info['confidence']*100:.1f}%"
                        )
                        del self.learning_notifications[user_id]
                        break
                await asyncio.sleep(10)  # Check every 10 seconds
                
        except Exception as e:
            logging.error(f"Error monitoring learning progress: {e}")
            await update.message.reply_text(
                f"Sorry, I lost track of the learning progress for {topic}."
            )

    async def _get_relevant_learned_context(self, message: str) -> str:
        """Get relevant learned context for the message"""
        try:
            # Get topic summary from learning service
            relevant_info = await self.learning_service.learner.get_topic_summary(message)
            if relevant_info and 'error' not in relevant_info:
                return f"Relevant learned context: {relevant_info}"
        except Exception as e:
            logging.error(f"Error getting learned context: {e}")
        return ""

    async def _get_topic_summary(self, topic: str) -> Optional[Dict]:
        """Get summary of learned topic"""
        try:
            summary = await self.learning_service.learner.get_topic_summary(topic)
            if summary and 'error' not in summary:
                return {
                    'summary': '\n'.join([
                        '• ' + point for point in summary['key_points'][:5]
                    ]),
                    'confidence': summary.get('confidence', 0.0)
                }
        except Exception as e:
            logging.error(f"Error getting topic summary: {e}")
        return None

    def setup_handlers(self):
        """Set up message handlers"""
        self.app.add_handler(CommandHandler("start", self.start_command))
        self.app.add_handler(CommandHandler("help", self.help_command))
        self.app.add_handler(CommandHandler("learn", self.learn_command))
        self.app.add_handler(CommandHandler("status", self.status_command))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))