"""Resumi – FastAPI application with Gradio UI."""

from functools import lru_cache
from typing import cast

import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel, Field

from resumi.config import get_settings
from resumi.core.agent import Agent
from resumi.core.embedding import (
    FaissKnowledgeBase,
    LocalHashEmbedder,
    OpenAITextEmbedder,
)
from resumi.core.gmail_handler import GmailHandler
from resumi.ui.gradio_ui import create_gradio_blocks
from resumi.core.mail_loader import MailLoader
from resumi.core.document_loader import DocumentLoader
# ---------------------------------------------------------------------------
# Dependency wiring (cached singletons)
# ---------------------------------------------------------------------------


@lru_cache
def _embedder() -> LocalHashEmbedder | OpenAITextEmbedder:
    s = get_settings()
    fallback = LocalHashEmbedder(dimension=s.openai_embedding_dimensions)
    if not s.openai_api_key or s.openai_api_key == "replace-me":
        return fallback
    return OpenAITextEmbedder(
        api_key=s.openai_api_key,
        model=s.openai_embedding_model,
        base_url=s.openai_base_url,
        dimensions=s.openai_embedding_dimensions,
        batch_size=s.openai_embedding_batch_size,
        fallback=fallback,
    )


@lru_cache
def _kb() -> FaissKnowledgeBase:
    s = get_settings()
    return FaissKnowledgeBase(
        docs_root=s.docs_root,
        index_root=s.faiss_index_dir,
        embedder=_embedder(),
        chunk_size=s.rag_chunk_size,
        chunk_overlap=s.rag_chunk_overlap,
        candidate_pool=s.rag_candidate_pool,
    )


@lru_cache
def _agent() -> Agent:
    s = get_settings()
    return Agent(
        knowledge_base=_kb(),
        model=s.openai_model,
        api_key=s.openai_api_key,
        base_url=s.openai_base_url,
        search_limit=s.rag_search_limit,
    )


@lru_cache
def _gmail() -> GmailHandler:
    s = get_settings()
    return GmailHandler(
        client_secrets_file=s.gmail_client_secrets_file,
        token_file=s.gmail_token_file,
        docs_root=s.docs_root,
        knowledge_base=_kb(),
        query=s.gmail_query,
        user_id=s.gmail_user_id,
    )
@lru_cache
def _mail_loader() -> MailLoader:
    s = get_settings()
    return MailLoader(docs_root=s.docs_root)
@lru_cache
def _document_loader() -> DocumentLoader:
    s = get_settings()
    return DocumentLoader(docs_root=s.docs_root, knowledge_base=_kb())
# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)


class GmailSyncRequest(BaseModel):
    max_results: int = Field(default=1000, ge=1, le=1000)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name)

    # -- routes --------------------------------------------------------------

    @app.get("/api/v1/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/v1/chat")
    async def chat(req: ChatRequest) -> dict[str, object]:
        return await _agent().chat(message=req.message)

    @app.post("/api/v1/gmail/sync")
    async def gmail_sync(req: GmailSyncRequest) -> dict[str, object]:
        half = req.max_results // 2
        return await _gmail().sync(received_max=half, sent_max=req.max_results - half)

    # -- Gradio UI -----------------------------------------------------------

    blocks = create_gradio_blocks(
    agent=_agent(),
    mail_loader=_mail_loader(),
    document_loader=_document_loader(),
    )
    return cast(FastAPI, gr.mount_gradio_app(app, blocks, path=settings.gradio_path))


app = create_app()
