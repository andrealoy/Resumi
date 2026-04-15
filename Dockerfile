FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps required by pymupdf / faiss-cpu / sounddevice
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libmupdf-dev libfreetype6 libharfbuzz0b libjpeg62-turbo \
        libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN uv sync --no-dev --frozen || uv sync --no-dev

# Prepare runtime directories
RUN mkdir -p docs .data/faiss credentials

COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
CMD ["uv", "run", "uvicorn", "resumi.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "src"]
