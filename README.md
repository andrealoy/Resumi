# Resumi

Personal assistant API built with FastAPI. Ingests Gmail messages, indexes them in a local FAISS knowledge base, and exposes a RAG-powered Gradio chat interface.

## Structure

```
src/resumi/
├── main.py              # FastAPI + routes + wiring
├── config.py            # Settings (.env)
├── core/
│   ├── agent.py         # RAG chat agent
│   ├── embedding.py     # Embedders + FAISS index
│   ├── gmail_handler.py # Gmail sync
│   ├── audio_handler.py # Placeholder
│   ├── tools.py         # Placeholder
│   └── calendar.py      # Placeholder
└── ui/
    ├── gradio_ui.py     # Gradio UI
    └── chat.py          # UI ↔ Agent bridge
```

## Quick start

```bash
cp .env.example .env     # edit with your OpenAI key
uv sync --all-groups
uv run uvicorn resumi.main:app --reload --app-dir src
```

Open `http://127.0.0.1:8000/gradio`.

## Gmail setup

1. Create an OAuth Desktop app in [Google Cloud Console](https://console.cloud.google.com/).
2. Download the client JSON → `credentials/gmail-client-secret.json`.
3. Call `POST /api/v1/gmail/sync` — your browser opens for consent.
4. Token saved to `credentials/gmail-token.json`.

```bash
curl -X POST http://127.0.0.1:8000/api/v1/gmail/sync \
  -H "Content-Type: application/json" \
  -d '{"max_results": 200}'
```

## API

| Method | Endpoint           | Description  |
|--------|--------------------|--------------|
| GET    | /api/v1/health     | Health check |
| POST   | /api/v1/chat       | RAG chat     |
| POST   | /api/v1/gmail/sync | Sync Gmail   |

## Dev

```bash
make lint    # ruff + mypy
make test    # pytest
make run     # uvicorn --reload
```

## Linting

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

## Deployment

GitHub Actions workflows are included for:

- Continuous integration on pull requests and pushes.
- Docker image publishing to Amazon ECR.
- Amazon ECS deployment using a task definition template.

Update the AWS secrets and ECS metadata in the workflow files before using the deployment pipeline.
