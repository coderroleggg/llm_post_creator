"""
Telegram bot for processing audio files and voice messages.
"""

import asyncio
import logging
from pathlib import Path
import json
from typing import Dict, List, Optional

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from openai import OpenAI
from pydantic_settings import BaseSettings
from telegram import InputFile, Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

DEFAULT_PROMPT = """
Ты - профессиональный SMM-специалист в телеграм канале

Вот примеры постов в канале:
{examples}

Вот сырой текст идеи с диктофона для следующего поста:
{text}

Преврати сырой текст в качественный телеграм пост в моем стиле, опираясь на примеры постов в канале. 
В ответе напиши только сам пост. Если нужно форматировать текст, используй markdown.
"""

MAX_FILE_SIZE_MB = 100  # Maximum allowed file size in megabytes
MAX_TELEGRAM_TEXT_LENGTH = 4096  # Telegram's maximum message length
MAX_EXAMPLE_POSTS = 10  # Maximum number of example posts to include in the prompt


class Settings(BaseSettings):
    """Application settings."""

    TELEGRAM_BOT_TOKEN: str
    ELEVENLABS_API_KEY: str
    OPENAI_API_KEY: str
    LLM_MODEL: str
    DEFAULT_PROMPT: str = DEFAULT_PROMPT
    LLM_BASE_URL: str = ""
    TEMP_DIR: Path = Path("temp")
    USER_DATA_FILE: Path = Path("user_data.json")
    EXAMPLE_POSTS_COUNT: int = MAX_EXAMPLE_POSTS

    class Config:
        env_file = ".env"


class UserChannelData:
    """Store user example posts data."""

    def __init__(self, data_file: Path):
        self.data_file = data_file
        self.user_examples: Dict[str, List[str]] = {}
        self.load_data()

    def load_data(self):
        """Load user data from file."""
        if self.data_file.exists():
            try:
                with open(self.data_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Поддержка обратной совместимости - конвертация старого формата в новый
                    for user_id, value in data.items():
                        if isinstance(value, str):  # Старый формат - channel_id
                            self.user_examples[user_id] = []
                        else:  # Новый формат - список примеров
                            self.user_examples[user_id] = value
            except json.JSONDecodeError:
                self.user_examples = {}
        else:
            self.user_examples = {}

    def save_data(self):
        """Save user data to file."""
        with open(self.data_file, "w", encoding="utf-8") as f:
            json.dump(self.user_examples, f, ensure_ascii=False, indent=4)

    def add_example_posts(self, user_id: str, posts: List[str]):
        """Add example posts for user."""
        if user_id not in self.user_examples:
            self.user_examples[user_id] = []
        
        # Добавляем новые примеры и ограничиваем их количество
        self.user_examples[user_id].extend(posts)
        if len(self.user_examples[user_id]) > MAX_EXAMPLE_POSTS:
            self.user_examples[user_id] = self.user_examples[user_id][-MAX_EXAMPLE_POSTS:]
        
        self.save_data()
        return len(self.user_examples[user_id])
    
    def clear_example_posts(self, user_id: str):
        """Clear all example posts for user."""
        self.user_examples[user_id] = []
        self.save_data()
    
    def get_example_posts(self, user_id: str) -> List[str]:
        """Get example posts for user."""
        return self.user_examples.get(user_id, [])


class VoiceAssistantBot:
    """Telegram bot for processing audio files and voice messages."""

    def __init__(self):
        """Initialize the bot and its dependencies."""
        load_dotenv()
        self.settings = Settings()
        self.setup_logging()
        self.create_temp_directory()
        
        # Initialize user channel data
        self.user_channel_data = UserChannelData(self.settings.USER_DATA_FILE)

        # Create API clients
        self.elevenlabs_client = ElevenLabs(api_key=self.settings.ELEVENLABS_API_KEY)
        self.openai_client = OpenAI(api_key=self.settings.OPENAI_API_KEY, base_url=self.settings.LLM_BASE_URL or None)

    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

    def create_temp_directory(self):
        """Create temporary directory for downloaded files."""
        self.settings.TEMP_DIR.mkdir(exist_ok=True)
        self.logger.info(f"Temporary directory: {self.settings.TEMP_DIR}")

    def run(self):
        """Start the bot."""
        self.logger.info("Starting bot initialization")

        # Create the Application
        application = Application.builder().token(self.settings.TELEGRAM_BOT_TOKEN).build()

        # Add handlers
        application.add_handler(CommandHandler("start", self.start))
        application.add_handler(CommandHandler("help", self.help_command))
        application.add_handler(CommandHandler("examples", self.examples_command))
        application.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, self.handle_audio))
        application.add_handler(MessageHandler(filters.FORWARDED, self.handle_forwarded_message))

        # Start the Bot
        self.logger.info("Bot is starting...")
        application.run_polling()

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send a message when the command /start is issued."""
        self.logger.info(f"Start command received from user {update.effective_user.id}")

        # Get current configuration
        model = self.settings.LLM_MODEL
        prompt = self.settings.DEFAULT_PROMPT
        base_url = self.settings.LLM_BASE_URL or "Default OpenAI"
        
        # Get user's examples count
        user_id = str(update.effective_user.id)
        example_posts = self.user_channel_data.get_example_posts(user_id)
        example_info = f"- Сохраненных примеров постов: {len(example_posts)}/{MAX_EXAMPLE_POSTS}" if example_posts else "- Нет сохраненных примеров постов"

        await update.message.reply_text(
            "Привет! Я бот-ассистент по созданию постов. Отправьте мне аудиофайл или голосовое сообщение, "
            "и я расшифрую его и структурирую для вас!\n\n"
            "Доступные команды:\n"
            "/start - Запустить бота\n"
            "/help - Показать это сообщение\n"
            "/examples - Управление примерами постов\n\n"
            f"Текущие настройки:\n"
            f"{example_info}\n\n"
            f"Чтобы добавить примеры постов, просто перешлите сообщения из вашего канала."
        )

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send a message when the command /help is issued."""
        self.logger.info(f"Help command received from user {update.effective_user.id}")
        await self.start(update, context)  # Reuse the start message
    
    async def examples_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /examples command to manage example posts."""
        user_id = str(update.effective_user.id)
        
        # Проверяем, есть ли аргументы команды
        if context.args and context.args[0].lower() == "clear":
            # Очищаем примеры пользователя
            self.user_channel_data.clear_example_posts(user_id)
            await update.message.reply_text(
                "🗑️ Все примеры постов удалены!\n\n"
                "Чтобы добавить новые примеры, просто перешлите сообщения из вашего канала."
            )
            return
        
        # Получаем текущие примеры пользователя
        examples = self.user_channel_data.get_example_posts(user_id)
        example_count = len(examples)
        
        # Формируем сообщение в зависимости от наличия примеров
        if example_count > 0:
            await update.message.reply_text(
                f"📊 У вас сохранено {example_count} примеров постов.\n\n"
                f"Чтобы добавить еще примеры, просто перешлите сообщения из вашего канала.\n"
                f"Чтобы удалить все примеры, используйте команду /examples clear"
            )
        else:
            await update.message.reply_text(
                "📝 У вас еще нет сохраненных примеров постов.\n\n"
                "Чтобы добавить примеры, просто перешлите сюда сообщения из вашего канала.\n"
                "Эти примеры будут использоваться для определения стиля ваших постов."
            )

    async def handle_forwarded_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle forwarded messages as example posts."""
        user_id = str(update.effective_user.id)
        
        # Определяем тип сообщения для логирования
        message_type = "unknown"
        if update.message.photo:
            message_type = "photo"
        elif update.message.video:
            message_type = "video"
        elif update.message.audio:
            message_type = "audio"
        elif update.message.voice:
            message_type = "voice"
        elif update.message.sticker:
            message_type = "sticker"
        elif update.message.document:
            message_type = "document"
        elif update.message.text and not update.message.photo and not update.message.video and not update.message.audio:
            message_type = "text"
        
        self.logger.info(f"Received forwarded {message_type} message from user {update.effective_user.id}")
        
        # Извлекаем текст из сообщения, если он есть
        post_text = update.message.text or update.message.caption or ""
        
        if post_text.strip():  # Проверяем, что текст не пустой после удаления пробелов
            # Добавляем пример в хранилище
            total_examples = self.user_channel_data.add_example_posts(user_id, [post_text])
            
            # Сообщаем пользователю о добавлении примера
            if message_type == "text":
                success_message = "✅ Пример поста сохранен!"
            else:
                success_message = f"✅ Текст из {message_type}-сообщения сохранен как пример!"
                
            await update.message.reply_text(
                f"{success_message}\n"
                f"Всего сохранено примеров: {total_examples}/{MAX_EXAMPLE_POSTS}\n\n"
                f"Вы можете продолжать пересылать сообщения из канала, чтобы добавить больше примеров.\n"
                f"Используйте команду /examples для управления примерами."
            )
        else:
            # В сообщении нет текста или подписи
            await update.message.reply_text(
                "⚠️ В пересланном сообщении нет текста или подписи. "
                "Пожалуйста, пересылайте сообщения, содержащие текст."
            )

    # Retry logic for ElevenLabs and OpenAI API calls
    @retry(
        stop=stop_after_attempt(3),  # Retry up to 3 times
        wait=wait_fixed(2),  # Wait 2 seconds between retries
        retry=retry_if_exception_type(Exception),  # Retry only on exceptions
    )
    async def transcribe_audio(self, audio_path: Path) -> str:
        """Transcribe audio using ElevenLabs."""
        self.logger.info("Starting transcription with ElevenLabs")

        # Run blocking operations in a separate thread pool to avoid blocking the event loop
        def _read_file_and_transcribe():
            with open(audio_path, "rb") as audio_file:
                audio_data = audio_file.read()
                return self.elevenlabs_client.speech_to_text.convert(
                    file=audio_data, model_id="scribe_v1", tag_audio_events=False, diarize=False
                )

        # Run the blocking operation in an executor
        result = await asyncio.get_event_loop().run_in_executor(None, _read_file_and_transcribe)

        self.logger.info("Successfully transcribed audio")
        return result.text

    async def download_file(self, file_id: str, bot) -> Path:
        """Download a file from Telegram."""
        self.logger.info(f"Starting download of file {file_id}")
        file = await bot.get_file(file_id)

        local_path = self.settings.TEMP_DIR / f"{file_id}"
        self.logger.info(f"Downloading file to {local_path}")

        try:
            await file.download_to_drive(local_path)
            self.logger.info(f"Successfully downloaded file to {local_path}")
            return local_path
        except Exception as e:
            self.logger.error(f"Error downloading file {file_id}: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),  # Retry up to 3 times
        wait=wait_fixed(2),  # Wait 2 seconds between retries
        retry=retry_if_exception_type(Exception),  # Retry only on exceptions
    )
    async def structure_text(self, text: str, context: ContextTypes.DEFAULT_TYPE, user_id: str) -> str:
        """Structure the text using OpenAI."""
        self.logger.info("Starting text structuring with OpenAI")

        # Get model and prompt from user preferences or use defaults
        model = self.settings.LLM_MODEL
        prompt = self.settings.DEFAULT_PROMPT
        
        # Get example posts for the user
        examples = ""
        example_posts = self.user_channel_data.get_example_posts(user_id)
        
        if example_posts:
            examples = "\n\n".join(example_posts)
            self.logger.info(f"Added {len(example_posts)} example posts to prompt for user {user_id}")
        else:
            self.logger.info(f"User {user_id} has no example posts")
            examples = "No examples available"

        final_prompt = prompt.format(text=text, examples=examples)
        self.logger.info(f"Final prompt: {final_prompt}")

        def _call_openai():
            return self.openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": final_prompt}],
            )

        # Run the blocking operation in an executor
        response = await asyncio.get_event_loop().run_in_executor(None, _call_openai)

        structured_text = response.choices[0].message.content
        self.logger.info("Successfully structured text with OpenAI")
        return structured_text

    def escape_markdown(self, text: str) -> str:
        markdown_special_chars = [
            '\\', '`', '*', '_', '{', '}', '[', ']', '(', ')',
            '#', '+', '-', '.', '!', '>'
        ]
        for char in markdown_special_chars:
            text = text.replace(char, f'\\{char}')
        return text

    async def send_text_or_file(self, update: Update, text: str, file_name: str, file_extension: str):
        """Send text or file depending on its length."""
        if len(text) > MAX_TELEGRAM_TEXT_LENGTH:
            # If text is too long, send as a file
            filename = f"{file_name}.{file_extension}"
            # Create InputFile from text string directly
            input_file = InputFile(
                obj=text,  # Text will be automatically encoded to bytes
                filename=filename,
            )
            await update.message.reply_document(document=input_file)
        else:
            
            await update.message.reply_text(self.escape_markdown(text), parse_mode=ParseMode.MARKDOWN_V2)

    async def handle_audio(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming audio files and voice messages."""
        self.logger.info(f"Received audio message from user {update.effective_user.id}")
        local_path = None
        status_message = None
        user_id = str(update.effective_user.id)

        try:
            # Determine the type of audio message
            if update.message.voice:
                file_id = update.message.voice.file_id
                self.logger.info("Processing voice message")
            elif update.message.audio:
                file_id = update.message.audio.file_id
                self.logger.info("Processing audio file")
            else:
                self.logger.warning("Received message without audio content")
                await update.message.reply_text("Please send an audio file or voice message.")
                return

            # Get file size and check if it exceeds the limit
            file_info = await context.bot.get_file(file_id)
            if file_info.file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
                await update.message.reply_text(f"File size exceeds the limit of {MAX_FILE_SIZE_MB} MB.")
                return

            # Send initial status message
            status_message = await update.message.reply_text("📥 Downloading your audio file...")

            # Download the file
            local_path = await self.download_file(file_id, context.bot)

            # Update status message
            await status_message.edit_text("🔍 Transcribing audio using ElevenLabs...")

            # Transcribe the audio
            transcription = await self.transcribe_audio(local_path)
            await self.send_text_or_file(update, transcription, f"transcription_{file_id}", "txt")
            self.logger.info("Sent raw transcription to user")

            # Update status message
            await status_message.edit_text("🧠 Structuring text using AI model...")

            # Structure the text
            structured_text = await self.structure_text(transcription, context, user_id)
            await self.send_text_or_file(update, structured_text, f"structured_text_{file_id}", "md")
            self.logger.info("Sent structured text to user")

            # Final status update
            await status_message.edit_text("✅ Processing complete!")

        except Exception as e:
            self.logger.error(f"Error processing audio: {str(e)}", exc_info=True)
            error_message = "Sorry, I couldn't process that audio."

            # Update status message if it exists
            if status_message:
                await status_message.edit_text(f"❌ {error_message}")
            else:
                await update.message.reply_text(error_message)
        finally:
            # Clean up temporary files (non-blocking)
            asyncio.create_task(self.cleanup_temp_file(local_path))

    async def cleanup_temp_file(self, file_path):
        """Clean up temporary files asynchronously."""
        try:
            if file_path and file_path.exists():
                # Run blocking file operation in executor
                await asyncio.get_event_loop().run_in_executor(
                    None, lambda: file_path.unlink() if file_path.exists() else None
                )
                self.logger.info(f"Deleted temporary file {file_path}")
        except Exception as e:
            self.logger.error(f"Error cleaning up temporary files: {str(e)}")


def main() -> None:
    """Start the bot."""
    bot = VoiceAssistantBot()
    bot.run()


if __name__ == "__main__":
    main()
