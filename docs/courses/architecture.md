# Resumi Architecture

## Goal

Resumi is a simple personal assistant. It ingests Gmail messages, indexes them in a local FAISS vector database, and exposes a RAG-powered chat interface.

## Main flow

1. Gmail synchronization saves cleaned messages in `docs/from_gmail/{received,sent}/`.
2. Every saved document is chunked and indexed into a FAISS knowledge base.
3. The agent uses that corpus + OpenAI to answer questions via RAG.
4. A Gradio chat UI is mounted inside FastAPI.

## Project layout

```
src/resumi/
├── main.py           # FastAPI app + routes + dependency wiring
├── config.py         # Settings from .env (pydantic-settings)
├── core/
│   ├── agent.py      # RAG chat: query routing, search, OpenAI call
│   ├── embedding.py  # Embedders (OpenAI + hash fallback) + FAISS index
│   ├── gmail_handler.py  # Gmail OAuth, fetch, save markdown, reindex
│   ├── audio_handler.py  # Placeholder (Whisper, to implement)
│   ├── tools.py      # Placeholder: classify_mail(), add_to_calendar()
│   └── calendar.py   # Placeholder (Google Calendar)
└── ui/
    ├── gradio_ui.py  # Gradio Blocks interface
    └── chat.py       # Async bridge: UI ↔ Agent
```

## API endpoints

| Method | Path              | Description      |
|--------|-------------------|------------------|
| GET    | /api/v1/health    | Health check     |
| POST   | /api/v1/chat      | RAG chat         |
| POST   | /api/v1/gmail/sync| Sync Gmail       |

## Design choices

- Flat structure: every file maps directly to a feature.
- Storage is visible on disk (`docs/`), easy to inspect and debug.
- FAISS indexing is local and deterministic — no external vector DB needed.
- Hash-based fallback embedder when no OpenAI key is set.
- Hybrid search: semantic (FAISS) + lexical (token overlap) scoring.

## Next steps

- Implement audio transcription (Whisper) in `audio_handler.py`.
- Implement tool calls (`tools.py`, `calendar.py`) for the agent.
- Add background workers for Gmail sync and audio processing.
- Replace in-memory state with persistent storage.
