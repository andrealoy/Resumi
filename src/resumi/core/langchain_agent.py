"""LangChain helpers for Resumi: tool execution + lightweight routing."""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from resumi.core.tools import calendar_tool, calculator_tool, web_search_tool

logger = logging.getLogger(__name__)

_VALID_TOOLS = frozenset({
    "calc",
    "web",
    "calendar",
    "draft",
    "gmail_connect",
    "gmail_sync",
    "gmail_status",
    "list_mails",
    "classify_mails",
    "classify_docs",
    "none",
})


def build_langchain_agent(model: str, api_key: str):
    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        temperature=0,
    )

    agent = create_agent(
        model=llm,
        tools=[calculator_tool, web_search_tool, calendar_tool],
        system_prompt=(
            "Tu es Resumi, un assistant utile et concis. "
            "Tu peux utiliser des outils quand c'est nécessaire. "
            "Utilise calculator_tool pour les calculs. "
            "Utilise web_search_tool pour les informations récentes ou web. "
            "Utilise calendar_tool pour enregistrer un événement ou un rendez-vous. "
            "Quand tu appelles calendar_tool, donne une date/heure explicite au format français clair ou YYYY-MM-DD HH:MM, et conserve l'heure exacte demandée par l'utilisateur. "
            "Si aucun outil n'est nécessaire, réponds directement."
        ),
    )

    return agent


def build_tool_router(model: str, api_key: str) -> ChatOpenAI:
    """Small deterministic LLM used only to decide if a tool is needed."""
    return ChatOpenAI(model=model, api_key=api_key, temperature=0)


def route_tool_call(
    llm: ChatOpenAI,
    *,
    message: str,
    history: list[dict[str, str]] | None = None,
) -> str | None:
    """Return a tool name, or *None* to continue with normal RAG flow."""
    recent = []
    for item in (history or [])[-6:]:
        role = item.get("role", "user")
        content = item.get("content", "")
        recent.append(f"{role}: {content[:300]}")
    history_text = "\n".join(recent) or "(aucun historique)"

    result = llm.invoke(
        [
            (
                "system",
                "Tu es un routeur de tools pour Resumi. "
                "Choisis au maximum un seul tool utile. "
                "Si aucun tool n'est vraiment nécessaire, renvoie none. "
                "Pour les questions sur des documents personnels, le CV, "
                "les mails déjà synchronisés, ou les réponses générales, choisis none "
                "car le flux RAG normal s'en chargera. "
                "Réponds uniquement en JSON avec les clés tool et reason.",
            ),
            (
                "human",
                "Tools disponibles :\n"
                "- calc: calculs numériques\n"
                "- web: recherche web / infos récentes\n"
                "- calendar: ajout d'événement au calendrier\n"
                "- draft: brouillon de réponse à un mail\n"
                "- gmail_connect: connecter Gmail\n"
                "- gmail_sync: synchroniser Gmail\n"
                "- gmail_status: statut Gmail\n"
                "- list_mails: lister les mails\n"
                "- classify_mails: classer les mails\n"
                "- classify_docs: classer les documents\n"
                "- none: pas de tool, laisser la réponse normale / RAG\n\n"
                f"Historique récent :\n{history_text}\n\n"
                f"Message utilisateur :\n{message}\n\n"
                'Format strict attendu : {"tool":"none|calc|web|calendar|draft|gmail_connect|gmail_sync|gmail_status|list_mails|classify_mails|classify_docs","reason":"..."}',
            ),
        ]
    )

    text = _to_text(result.content)
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        payload = json.loads(text[start : end + 1])
    except Exception as exc:
        logger.warning("LLM router JSON parse failed: %s | raw=%s", exc, text[:300])
        return None

    tool = str(payload.get("tool", "none")).strip()
    if tool not in _VALID_TOOLS or tool == "none":
        return None
    return tool


def _to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif hasattr(item, "text") and isinstance(item.text, str):
                parts.append(item.text)
        return "\n".join(parts)
    return str(content)