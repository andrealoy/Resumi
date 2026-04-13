"""Tools exposed to the assistant."""

from __future__ import annotations
from langchain_core.tools import tool

import ast
import pandas as pd
import os
import dateparser
import operator as op
from typing import Callable

from resumi.core.web_search import web_search


_ALLOWED_OPS: dict[type, Callable] = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
}

@tool
def _safe_eval(node: ast.AST) -> float:
    """
    Effectue un calcul mathématique sécurisé.
    Exemple d'entrée : '2 + 2' ou '(10 * 5) / 2'.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)

    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return float(_ALLOWED_OPS[type(node.op)](left, right))

    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        operand = _safe_eval(node.operand)
        return float(_ALLOWED_OPS[type(node.op)](operand))

    raise ValueError("Expression non autorisée.")

@tool
def calculator_tool(expression: str) -> str:
    """Effectue un calcul simple à partir d'une expression mathématique."""
    try:
        parsed = ast.parse(expression, mode="eval")
        result = _safe_eval(parsed.body)
        if result.is_integer():
            return str(int(result))
        return str(result)
    except Exception as exc:
        return f"Erreur de calcul : {exc}"

@tool
def web_search_tool(query: str) -> str:
    """Recherche une information récente sur le web."""
    try:
        return web_search(query)
    except Exception as exc:
        return f"Erreur de recherche web : {exc}"

@tool
def rag_tool(agent, question: str) -> str:
    """Répond à une question à partir des documents internes indexés."""
    route = agent._route(question)
    sources = agent._search(question, route)

    if not sources:
        return (
            "Aucun document pertinent n'a été trouvé dans la base interne."
        )

    context = agent._build_context(sources, route)
    citations = "\n".join(
        f"- {s.relative_path} (chunk {s.chunk_index})"
        for s in sources[:5]
    )

    return f"{context}\n\nCitations possibles:\n{citations}"



CALENDAR_FILE = "local_calendar.csv"

@tool
def calendar_tool(event_details: str, date_time: str) -> str:
    """Enregistre un événement dans le calendrier. 
    'date_time' peut être un format ISO ou une date naturelle en français."""
    try:
        # dateparser va tenter de comprendre des dates de formats moins précis comme "jeudi 13"
        dt = dateparser.parse(date_time, languages=['fr'], settings={'PREFER_DATES_FROM': 'future'})
        
        if dt is None:
            return "ERREUR : Je n'ai pas réussi à identifier la date à laquelle tu fais référence. Peux-tu être plus précis ? :)"

        if dt.hour == 0 and dt.minute == 0:
            dt = dt.replace(hour=9, minute=0)
            duree = " (09:00 - 10:00)"
        else:
            duree = ""
        
        new_event = {
            "Date": dt.strftime("%Y-%m-%d"),
            "Heure": dt.strftime("%H:%M"),
            "Événement": event_details
        }
        
        # Sauvegarde locale (CSV)
        df = pd.read_csv(CALENDAR_FILE) if os.path.exists(CALENDAR_FILE) else pd.DataFrame()
        df = pd.concat([df, pd.DataFrame([new_event])], ignore_index=True)
        df = df.sort_values(by=["Date", "Heure"]) # On trie par date
        df.to_csv(CALENDAR_FILE, index=False)
        
        return f"SUCCÈS : '{event_details}' ajouté pour le {dt.strftime('%d/%m/%Y à %H:%M')}{duree}."
    except Exception as e:
        return f"ERREUR : {str(e)}"