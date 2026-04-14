"""Minimal LangChain agent for Resumi."""

from __future__ import annotations

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from resumi.core.tools import calculator_tool, web_search_tool, calendar_tool


def build_langchain_agent(model: str, api_key: str):
    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
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
            "Si aucun outil n'est nécessaire, réponds directement."
        ),
    )

    return agent