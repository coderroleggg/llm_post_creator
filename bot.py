"""
Telegram bot for processing audio files and voice messages.
"""

import asyncio
import logging
from pathlib import Path

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from openai import OpenAI
from pydantic_settings import BaseSettings
from telegram import InputFile, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

DEFAULT_PROMPT = """
You are an expert in processing speech transcriptions. Analyze the following audio transcription and transform it into clear, well-structured information. Follow these principles:

1. Identify the main topic and key points
2. Group related ideas into logical sections with headings
3. Fix inaccuracies, repetitions, and conversational elements
4. Highlight important facts, figures, and data
5. Use bullet points for enumerations
6. Employ concise, clear phrasing
7. Preserve important quotes in quotation marks
8. Organize information by priority or chronology
9. Add a brief conclusion at the end
10. Respond in the same language as the transcription

Transcription:
{text}
"""

MAX_FILE_SIZE_MB = 100  # Maximum allowed file size in megabytes
MAX_TELEGRAM_TEXT_LENGTH = 4096  # Telegram's maximum message length


class Settings(BaseSettings):
    """Application settings."""

    TELEGRAM_BOT_TOKEN: str
    ELEVENLABS_API_KEY: str
    OPENAI_API_KEY: str
    LLM_MODEL: str
    DEFAULT_PROMPT: str = DEFAULT_PROMPT
    LLM_BASE_URL: str = ""
    TEMP_DIR: Path = Path("temp")

    class Config:
        env_file = ".env"


class VoiceAssistantBot:
    """Telegram bot for processing audio files and voice messages."""

    def __init__(self):
        """Initialize the bot and its dependencies."""
        load_dotenv()
        self.settings = Settings()
        self.setup_logging()
        self.create_temp_directory()

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
        application.add_handler(CommandHandler("model", self.model_command))
        application.add_handler(CommandHandler("prompt", self.prompt_command))
        application.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, self.handle_audio))

        # Start the Bot
        self.logger.info("Bot is starting...")
        application.run_polling()

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send a message when the command /start is issued."""
        self.logger.info(f"Start command received from user {update.effective_user.id}")

        # Get current configuration
        model = context.user_data.get("model", self.settings.LLM_MODEL)
        prompt = context.user_data.get("prompt", self.settings.DEFAULT_PROMPT)
        base_url = self.settings.LLM_BASE_URL or "Default OpenAI"

        await update.message.reply_text(
            "Hi! I'm your voice assistant bot. Send me an audio file or voice message, "
            "and I'll transcribe it and structure it for you!\n\n"
            "Available commands:\n"
            "/start - Start the bot\n"
            "/help - Show this help message\n"
            "/model <model_name> - Change the LLM model (e.g., /model gpt-4-turbo)\n"
            "/prompt <prompt_text> - Change the prompt template (e.g., /prompt Summarize this text:\n\n{text})\n\n"
            f"Current settings:\n"
            f"- LLM Model: {model}\n"
            f"- LLM Base URL: {base_url}\n"
            f"- Default Prompt: {prompt}"
        )

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send a message when the command /help is issued."""
        self.logger.info(f"Help command received from user {update.effective_user.id}")
        await self.start(update, context)  # Reuse the start message

    async def model_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /model command to change the LLM model."""
        if not context.args:
            await update.message.reply_text("Please specify a model. Example: /model gpt-4-turbo")
            return

        model = context.args[0]
        context.user_data["model"] = model
        await update.message.reply_text(f"Model changed to: {model}")

    async def prompt_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /prompt command to change the prompt."""
        if not context.args:
            await update.message.reply_text("Please specify a prompt. Example: /prompt Summarize this text:\n\n{text}")
            return

        prompt = " ".join(context.args)
        context.user_data["prompt"] = prompt
        await update.message.reply_text(f"Prompt changed to: {prompt}")

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
    async def structure_text(self, text: str, context: ContextTypes.DEFAULT_TYPE) -> str:
        """Structure the text using OpenAI."""
        self.logger.info("Starting text structuring with OpenAI")

        # Get model and prompt from user preferences or use defaults
        model = context.user_data.get("model", self.settings.LLM_MODEL)
        prompt = context.user_data.get("prompt", self.settings.DEFAULT_PROMPT)

        # Run blocking API call in a separate thread
        def _call_openai():
            return self.openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt.format(text=text)}],
            )

        # Run the blocking operation in an executor
        response = await asyncio.get_event_loop().run_in_executor(None, _call_openai)

        structured_text = response.choices[0].message.content
        self.logger.info("Successfully structured text with OpenAI")
        return structured_text

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
            # If text is short, send as a message and as a file
            await update.message.reply_text(text)

            # Also send as a document
            filename = f"{file_name}.{file_extension}"
            input_file = InputFile(
                obj=text,  # Text will be automatically encoded to bytes
                filename=filename,
            )
            await update.message.reply_document(document=input_file)

    async def handle_audio(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming audio files and voice messages."""
        self.logger.info(f"Received audio message from user {update.effective_user.id}")
        local_path = None
        status_message = None

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
            structured_text = await self.structure_text(transcription, context)
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
