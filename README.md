# Resumi

Personal assistant API built with FastAPI. Ingests Gmail messages, indexes them in a local FAISS knowledge base with SQLite metadata catalogue, and exposes a RAG-powered Gradio chat interface.

## Structure

```
src/resumi/
├── main.py               # FastAPI app factory, API routes, singleton wiring
├── config.py             # Pydantic Settings (.env)
├── core/
│   ├── agent.py          # RAG chat agent (routing → FAISS search → LLM)
│   ├── embedding.py      # Embedders + FAISS index (md/txt/pdf)
│   ├── gmail_handler.py  # Gmail OAuth + fetch/save/sync
│   ├── mail_loader.py    # Read synced mail markdown files
│   ├── mail_tools.py     # LLM tools: classify email & draft reply
│   ├── document_store.py # SQLite metadata catalogue
│   ├── document_loader.py# User file uploads + dedup + reindex
│   ├── audio_handler.py  # Placeholder
│   └── calendar.py       # Placeholder
└── ui/
    ├── gradio_ui.py      # Gradio UI (Chat + Documents + Mails)
    └── chat.py           # Sync/async bridge (Gradio ↔ Agent)
```

### Storage layout

```
docs/                       ← all documents (source of truth)
  from_gmail/received/      ← synced received mails (.md)
  from_gmail/sent/          ← synced sent mails (.md)
  <user_folders>/           ← uploaded docs (pdf, md, txt…)
.data/
  faiss/                    ← FAISS index + chunk metadata (.json)
  documents.db              ← SQLite catalogue (dedup, categories, dates)
credentials/
  gmail-client-secret.json  ← OAuth Desktop app credentials
  gmail-token.json          ← OAuth token (auto-generated)
```

## Quick start

```bash
cp .env.example .env     # edit with your OpenAI key
make install
make run
```

Open `http://127.0.0.1:8000` (redirects to `/gradio`).

## How it works

### Chat flow

1. User asks a question in the **Chat** tab.
2. `Agent.chat()` **routes** the query (keyword detection → sent / received / all).
3. **FAISS search** — semantic + lexical scoring, filtered by mail direction if needed.
4. If the question is **temporal** ("dernier mail", "ce message"…), the agent also queries **SQLite** for the most recent mails sorted by date and reads their file content.
5. The **conversation history** (last 10 turns) is included for follow-up questions.
6. Everything is sent to the **LLM** (OpenAI) which generates the answer.

### Gmail sync flow

1. Connect via OAuth → `credentials/gmail-token.json`.
2. Fetch received + sent messages via the Gmail API.
3. Save as markdown in `docs/from_gmail/`, dedup by content hash (SQLite).
4. Rebuild FAISS index (all docs: md, txt, pdf).
5. Auto-classify uncategorized mails by title via LLM → category persisted in SQLite.

Sync triggers **automatically** after connecting Gmail and on page load if the last sync is older than 24 h.

### Document upload

1. Upload files in the **Documents** tab (any text or PDF file).
2. Files are saved to `docs/<folder_name>/`, deduped via content hash.
3. FAISS index is rebuilt to include the new files.
4. Click **Classifier** to auto-classify all uploaded docs via LLM.

## Gmail setup

1. Create an OAuth Desktop app in [Google Cloud Console](https://console.cloud.google.com/).
2. Download the client JSON → `credentials/gmail-client-secret.json`.
3. Open the Gradio UI — click **Connecter Gmail** → browser opens for consent.
4. Token is saved; sync starts automatically.

To share with others: add their Gmail address as a **test user** in the OAuth consent screen (up to 100 users without app verification).

Manual sync via API:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/gmail/sync \
  -H "Content-Type: application/json" \
  -d '{"max_results": 200}'
```

## API

| Method | Endpoint             | Description            |
|--------|----------------------|------------------------|
| GET    | `/`                  | Redirect to `/gradio`  |
| GET    | `/api/v1/health`     | Health check           |
| POST   | `/api/v1/chat`       | RAG chat               |
| POST   | `/api/v1/gmail/sync` | Sync Gmail             |

## Environment variables

| Variable                     | Default                              | Description                     |
|------------------------------|--------------------------------------|---------------------------------|
| `OPENAI_API_KEY`             | `replace-me`                         | OpenAI API key                  |
| `OPENAI_MODEL`               | `gpt-5.4`                            | Chat model                      |
| `OPENAI_BASE_URL`            | —                                    | Custom API base URL             |
| `OPENAI_EMBEDDING_MODEL`     | `text-embedding-3-small`             | Embedding model                 |
| `OPENAI_EMBEDDING_DIMENSIONS`| `256`                                | Embedding dimensions            |
| `DOCS_ROOT`                  | `docs`                               | Documents directory             |
| `DB_PATH`                    | `.data/documents.db`                 | SQLite database path            |
| `FAISS_INDEX_DIR`            | `.data/faiss`                        | FAISS index directory           |
| `RAG_CHUNK_SIZE`             | `160`                                | Chunk size (tokens)             |
| `RAG_CHUNK_OVERLAP`          | `40`                                 | Chunk overlap                   |
| `RAG_SEARCH_LIMIT`           | `8`                                  | Max RAG results                 |
| `GMAIL_MAX_RESULTS`          | `100`                                | Mails fetched per direction     |

## Makefile

| Target         | Command                                        |
|----------------|------------------------------------------------|
| `make install` | `uv sync --all-groups`                         |
| `make lint`    | `ruff check` + `ruff format --check` + `mypy`  |
| `make test`    | `uv run pytest`                                |
| `make run`     | `uvicorn resumi.main:app --reload`             |

## Deployment

GitHub Actions workflows are included for:

- Continuous integration on pull requests and pushes.
- Docker image publishing to Amazon ECR.
- Amazon ECS deployment using a task definition template.

Update the AWS secrets and ECS metadata in the workflow files before using the deployment pipeline.
