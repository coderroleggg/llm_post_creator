# AI Voice Assistant

A Telegram bot that processes audio files and voice messages using ElevenLabs for transcription. This project provides an automated solution for converting voice messages and audio files into text using AI technology.

## Features

- Accepts audio files and voice messages
- Transcribes audio using ElevenLabs
- Docker support for easy deployment
- Environment-based configuration

## Installation

### Using Docker (Recommended)

1. Clone this repository:
```bash
git clone https://github.com/yourusername/ai_voice_assistant.git
cd ai_voice_assistant
```

2. Create a `.env` file:
```bash
cp .env.example .env
```

3. Edit the `.env` file and add your:
   - Telegram Bot Token (get it from @BotFather)
   - ElevenLabs API Key (get it from https://elevenlabs.io)

4. Start the bot using Docker Compose:
```bash
docker-compose up -d
```

### Manual Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/ai_voice_assistant.git
cd ai_voice_assistant
```

2. Create and activate a virtual environment:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
uv pip install -e .
```

4. Create and configure `.env` file as described above

## Usage

1. Start the bot:
```bash
# If using Docker
docker-compose up -d

# If running manually
python -m ai_voice_assistant.bot
```

2. In Telegram:
   - Send `/start` to initialize the bot
   - Send an audio file or voice message
   - The bot will process it and send back the transcription

## Development

The project uses:
- Python 3.12+
- UV for package management
- Ruff for linting
- Just for task automation

Common development tasks:
```bash
just fmt # Format code
```

## Requirements

- Python 3.12 or higher
- Telegram Bot Token
- ElevenLabs API Key
- Docker (optional, for containerized deployment)

## License

MIT
