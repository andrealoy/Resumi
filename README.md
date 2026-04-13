# Resumi

Personal assistant built with FastAPI + Gradio. Ingests Gmail messages, indexes them in a local FAISS knowledge base with SQLite metadata catalogue, and exposes a RAG-powered chat interface with intelligent routing.

## Features

- **Chat assistant** — keyword-routed agent with RAG, calculator, web search, draft-reply, and app-action tools
- **Intelligent search routing** — distinguishes between mail queries (sent/received), document queries (CV, diplomas) and general questions
- **Gmail sync** — OAuth connection, step-by-step progress in chat, configurable mail count
- **Document upload** — PDF, text, audio transcription via OpenAI Whisper
- **Auto-classification** — LLM-powered mail and document categorization
- **Voice input** — microphone recording + transcription in the chat
- **Dark UI** — violet-accented theme with Source Serif 4 typography

## Structure

```
src/resumi/
├── main.py                # FastAPI app factory, API routes, singleton wiring
├── config.py              # Pydantic Settings (.env)
├── core/
│   ├── agent.py           # RAG chat agent (tool routing → FAISS search → LLM)
│   ├── embedding.py       # Embedders + FAISS index (md/txt/pdf)
│   ├── gmail_handler.py   # Gmail OAuth + fetch/save/sync
│   ├── mail_loader.py     # Read synced mail markdown files
│   ├── mail_tools.py      # LLM tools: classify email & draft reply
│   ├── document_store.py  # SQLite metadata catalogue
│   ├── document_loader.py # User file uploads + dedup + reindex
│   ├── langchain_agent.py # LangChain agent (calculator + web search)
│   ├── audio_handler.py   # Microphone recording + OpenAI Whisper transcription
│   ├── tools.py           # Calculator tool (safe eval)
│   ├── web_search.py      # DuckDuckGo web search
│   └── calendar.py        # Placeholder
└── ui/
    ├── gradio_ui.py       # Gradio UI (Chat + Documents + Mails tabs)
    └── chat.py            # Sync/async bridge (Gradio ↔ Agent)
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

### Tool routing

`Agent._needs_tool()` classifies each message via keyword matching (substring + bag-of-words with hyphen splitting):

| Tool             | Examples                                              |
|------------------|-------------------------------------------------------|
| `calc`           | "calcule 100 puissance 15", "combien font 3 × 4"     |
| `web`            | "recherche sur le web", "dis-moi plus sur OpenAI"     |
| `draft`          | "rédige une réponse", "brouillon de mail"             |
| `gmail_connect`  | "connecte Gmail", "connecte-moi"                      |
| `gmail_sync`     | "synchronise mes mails", "sync"                       |
| `gmail_status`   | "statut Gmail", "Gmail est connecté ?"                |
| `classify_mails` | "classe mes mails", "trie mes mails"                  |
| `classify_docs`  | "classe mes documents"                                |
| `list_mails`     | "liste mes mails", "mes derniers mails"               |
| *(none → RAG)*   | Any other question → FAISS search + LLM               |

### Search routing

`Agent._route()` determines which documents to search:

| Route      | Trigger                              | Search scope                         |
|------------|--------------------------------------|--------------------------------------|
| `sent`     | "mails envoyés", "à qui ai-je écrit"| `from_gmail/sent/` only              |
| `received` | "mails reçus", "boite de réception" | `from_gmail/received/` only          |
| `docs`     | "mon CV", "compétences", "diplôme"  | Uploaded documents (excludes mails)  |
| `all`      | Everything else                      | All indexed documents                |

### Gmail sync flow

1. User asks "synchronise mes mails" in chat.
2. Agent asks how many mails to download (or user provides inline: "synchronise 200 mails").
3. Step-by-step progress displayed in chat (`[1/5]` … `[5/5]`):
   - Fetch received → fetch sent → save to disk → reindex FAISS → auto-classify.
4. Contextual confirmations ("vas y", "oui") also trigger sync after a connect prompt.

### Document upload

1. Upload files in the **Documents** tab (any text, PDF, or audio file).
2. Audio files are transcribed via OpenAI Whisper before indexing.
3. Files are saved to `docs/<folder_name>/`, deduped via content hash.
4. FAISS index is rebuilt to include the new files.
5. Click **Classifier** to auto-classify all uploaded docs via LLM.

## Gmail setup

1. Create an OAuth Desktop app in [Google Cloud Console](https://console.cloud.google.com/).
2. Download the client JSON → `credentials/gmail-client-secret.json`.
3. Open the Gradio UI — click **Connecter Gmail** → browser opens for consent.
4. Token is saved; you can then sync from the chat or the button.

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

| Variable                       | Default                              | Description                     |
|--------------------------------|--------------------------------------|---------------------------------|
| `OPENAI_API_KEY`               | `replace-me`                         | OpenAI API key                  |
| `OPENAI_MODEL`                 | `gpt-5.4`                            | Chat model                      |
| `OPENAI_BASE_URL`              | —                                    | Custom API base URL             |
| `OPENAI_EMBEDDING_MODEL`       | `text-embedding-3-small`             | Embedding model                 |
| `OPENAI_EMBEDDING_DIMENSIONS`  | `512`                                | Embedding vector dimensions     |
| `OPENAI_EMBEDDING_BATCH_SIZE`  | `32`                                 | Embedding API batch size        |
| `DOCS_ROOT`                    | `docs`                               | Documents directory             |
| `DB_PATH`                      | `.data/documents.db`                 | SQLite database path            |
| `FAISS_INDEX_DIR`              | `.data/faiss`                        | FAISS index directory           |
| `RAG_CHUNK_SIZE`               | `160`                                | Chunk size (tokens)             |
| `RAG_CHUNK_OVERLAP`            | `40`                                 | Chunk overlap                   |
| `RAG_SEARCH_LIMIT`             | `8`                                  | Max RAG results                 |
| `RAG_CANDIDATE_POOL`           | `24`                                 | FAISS candidate pool size       |
| `GMAIL_MAX_RESULTS`            | `100`                                | Default mails fetched per sync  |
| `LOG_LEVEL`                    | `INFO`                               | Logging level                   |

## Makefile

| Target           | Command                                        |
|------------------|------------------------------------------------|
| `make install`   | `uv sync --all-groups`                         |
| `make lint`      | `ruff check` + `ruff format --check` + `mypy`  |
| `make test`      | `uv run pytest`                                |
| `make agent-test`| `uv run pytest tests/test_tool_routing.py -v`  |
| `make run`       | `uvicorn resumi.main:app --reload`             |
| `make clean`     | Remove caches, token, FAISS, SQLite, docs      |

## Testing

```bash
make test          # all tests (health + routing)
make agent-test    # 104 keyword routing tests only
```

## Docker

```bash
docker build -t resumi .
docker run -p 8000:8000 --env-file .env -v ./credentials:/app/credentials resumi
```

## Deployment

GitHub Actions workflows are included for:

- Continuous integration on pull requests and pushes.
- Docker image publishing to Amazon ECR.
- Amazon ECS deployment using a task definition template.

Update the AWS secrets and ECS metadata in the workflow files before using the deployment pipeline.
