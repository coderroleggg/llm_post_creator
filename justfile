fmt:
    uv run ruff check --fix-only ai_voice_assistant
    uv run isort ai_voice_assistant
    uv run ruff format ai_voice_assistant

update-deps:
    uv pip compile --no-header --upgrade pyproject.toml -o requirements.txt

install:
    uv pip install -r requirements.txt

run:
    uv run python -m ai_voice_assistant
