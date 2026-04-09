FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml README.md ./
COPY src ./src

RUN uv sync --no-dev --frozen || uv sync --no-dev

COPY credentials ./credentials

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "resumi.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "src"]
