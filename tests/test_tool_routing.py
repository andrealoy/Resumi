"""Tests for Agent._needs_tool routing.

Run with: make agent-test
"""

import pytest

from resumi.core.agent import Agent


@pytest.fixture()
def agent():
    """Minimal Agent with no external deps — only _needs_tool is tested."""
    # _needs_tool is a pure method that only reads class-level keyword
    # tuples, so we can safely construct a lightweight instance.
    return Agent.__new__(Agent)


# ── Calculator ─────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "message",
    [
        "calcule 100 puissance 15",
        "combien font 3 fois 4",
        "combien fait 12 divisé par 3",
        "fais l'addition de 5 et 7",
        "soustraction de 10 et 3",
        "multiplication de 2 par 8",
        "10 puissance 50",
        "quelle est la racine de 144",
        "calcule le modulo de 17 par 5",
        "factorielle de 6",
    ],
)
def test_calc(agent: Agent, message: str):
    assert agent._needs_tool(message) == "calc"


# ── Web search ─────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "message",
    [
        "fais une recherche sur ArtFX",
        "cherche sur le web c'est quoi OpenAI",
        "dis m'en plus sur cette école",
        "dis-moi plus sur le campus de ArtFX",
        "recherche web sur les LLMs",
        "en savoir plus sur l'IA générative",
        "infos sur GPT-5",
        "informations sur le fondateur de SpaceX",
        "search the web for latest news",
        "cherche sur internet qui est Elon Musk",
    ],
)
def test_web(agent: Agent, message: str):
    assert agent._needs_tool(message) == "web"


# ── Draft reply ────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "message",
    [
        "génère un brouillon de réponse",
        "rédige une réponse à ce mail",
        "que répondre à ce message",
        "comment répondre à ce mail",
        "génère une réponse professionnelle",
        "écris un mail pour refuser poliment",
        "réponds à ce mail",
        "draft a reply",
        "genere une reponse",
    ],
)
def test_draft(agent: Agent, message: str):
    assert agent._needs_tool(message) == "draft"


# ── Gmail connect ──────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "message",
    [
        "connecte gmail",
        "connecte moi a gmail",
        "connecte-moi à gmail",
        "connexion gmail",
        "je veux me connecter à gmail",
        "login gmail",
        "connect gmail please",
        "je voudrais connecter gmail",
    ],
)
def test_gmail_connect(agent: Agent, message: str):
    assert agent._needs_tool(message) == "gmail_connect"


# ── Gmail sync ─────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "message",
    [
        "synchronise mes mails",
        "lance la synchro",
        "sync",
        "récupère mes mails",
        "charge mes mails",
        "importe mes mails s'il te plaît",
        "télécharge mes mails",
        "fetch mails",
        "fetch emails from gmail",
    ],
)
def test_gmail_sync(agent: Agent, message: str):
    assert agent._needs_tool(message) == "gmail_sync"


# ── Classify mails ─────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "message",
    [
        "classe mes mails",
        "classifie mes mails automatiquement",
        "catégorise mes mails",
        "trie mes mails par catégorie",
        "classify mails",
        "classe tous les mails non classés",
    ],
)
def test_classify_mails(agent: Agent, message: str):
    assert agent._needs_tool(message) == "classify_mails"


# ── Classify docs ──────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "message",
    [
        "classe mes documents",
        "classifie mes documents",
        "catégorise mes documents uploadés",
        "trie mes documents",
        "classe mes fichiers",
        "classify documents",
        "classify docs",
    ],
)
def test_classify_docs(agent: Agent, message: str):
    assert agent._needs_tool(message) == "classify_docs"


# ── Gmail status ───────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "message",
    [
        "statut gmail",
        "gmail est connecté ?",
        "status gmail",
        "état de gmail",
        "gmail status",
        "est-ce que gmail est connecte",
    ],
)
def test_gmail_status(agent: Agent, message: str):
    assert agent._needs_tool(message) == "gmail_status"


# ── List mails ─────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "message",
    [
        "liste mes mails",
        "montre mes mails",
        "affiche mes mails récents",
        "mes derniers mails",
        "list mails",
        "list emails",
        "show my mails",
        "affiche les mails reçus",
    ],
)
def test_list_mails(agent: Agent, message: str):
    assert agent._needs_tool(message) == "list_mails"


# ── RAG fallback (no tool should match) ────────────────────────────────

@pytest.mark.parametrize(
    "message",
    [
        "combien de crédits ECTS donne l'école ArtFX ?",
        "quel est le dernier mail reçu ?",
        "résume ce document",
        "bonjour",
        "c'est quoi ce projet Resumi ?",
    ],
)
def test_rag_fallback(agent: Agent, message: str):
    assert agent._needs_tool(message) is None


# ══════════════════════════════════════════════════════════════════════
# Route tests (_route: sent / received / docs / all)
# ══════════════════════════════════════════════════════════════════════

# ── Sent route ─────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "message",
    [
        "montre mes mails envoyés",
        "à qui ai-je envoyé un mail ?",
        "mail envoyé hier",
    ],
)
def test_route_sent(agent: Agent, message: str):
    route = agent._route(message)
    assert route["label"] == "sent"
    assert route["prefixes"] == ["from_gmail/sent/"]


# ── Received route ─────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "message",
    [
        "quels mails ai-je reçus ?",
        "boite de reception",
        "inbox",
        "qui m'a écrit ?",
    ],
)
def test_route_received(agent: Agent, message: str):
    route = agent._route(message)
    assert route["label"] == "received"
    assert route["prefixes"] == ["from_gmail/received/"]


# ── Docs route (user profile / CV / diplomas) ─────────────────────────

@pytest.mark.parametrize(
    "message",
    [
        "parle moi de mon CV",
        "quel poste tu me conseillerais ?",
        "quelles sont mes compétences ?",
        "combien d'années d'expérience j'ai ?",
        "quel est mon diplôme ?",
        "résume mon parcours pro",
        "mon profil correspond à quel emploi ?",
        "quelles sont mes formations ?",
        "attestation ECTS",
        "parle de mon cursus",
        "mes études c'était quoi ?",
        "mon portfolio",
        "mes skills",
        "quels postes me correspondent ?",
        "mes documents uploadés",
    ],
)
def test_route_docs(agent: Agent, message: str):
    route = agent._route(message)
    assert route["label"] == "docs", f"Expected docs for: {message}"
    assert route.get("exclude_prefixes") == ["from_gmail/"]


# ── All route (ambiguous / general) ───────────────────────────────────

@pytest.mark.parametrize(
    "message",
    [
        "bonjour",
        "c'est quoi ce projet Resumi ?",
        "résume ce document",
    ],
)
def test_route_all(agent: Agent, message: str):
    route = agent._route(message)
    assert route["label"] == "all"
