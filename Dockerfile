FROM python:3.12-slim-bullseye AS common-deps

ENV APP_DIR=/opt/project

WORKDIR ${APP_DIR}

COPY --from=ghcr.io/astral-sh/uv:0.6.3 /uv /uvx /bin/

COPY ./requirements.txt ${APP_DIR}/requirements.txt
RUN uv pip install --system --no-cache -r ${APP_DIR}/requirements.txt;

# -------------------- development dependencies and sources --------------
FROM common-deps AS dev

COPY ./pyproject.toml ./pyproject.toml
COPY ./ai_voice_assistant ./ai_voice_assistant

# -------------------- unit tests and linters --------------------
FROM dev AS dev-unittested

RUN isort --check ai_voice_assistant
RUN ruff format --check ai_voice_assistant
RUN ruff check ai_voice_assistant
RUN mypy ai_voice_assistant

ENTRYPOINT ["python", "-m", "ai_voice_assistant"]