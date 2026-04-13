"""RAG-powered assistant agent."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING

from openai import AsyncOpenAI

from resumi.core.document_store import DocumentStore
from resumi.core.embedding import DocumentMatch, FaissKnowledgeBase
from resumi.core.langchain_agent import build_langchain_agent
from resumi.core.mail_tools import classify_and_store as _classify_and_store
from resumi.core.mail_tools import classify_email as _classify_email
from resumi.core.mail_tools import draft_email_reply as _draft_email_reply

if TYPE_CHECKING:
    from resumi.core.gmail_handler import GmailHandler

logger = logging.getLogger(__name__)


class Agent:
    def __init__(
        self,
        *,
        knowledge_base: FaissKnowledgeBase,
        model: str,
        api_key: str,
        base_url: str | None = None,
        search_limit: int = 8,
        store: DocumentStore | None = None,
        client: AsyncOpenAI | None = None,
        gmail_handler: GmailHandler | None = None,
    ) -> None:
        self._kb = knowledge_base
        self._model = model
        self._search_limit = search_limit
        self._store = store
        self._gmail: GmailHandler | None = gmail_handler
        # Treat empty string as None so the SDK uses its default endpoint.
        # Also clean the env var: python-dotenv / uvicorn may export an empty
        # OPENAI_BASE_URL from .env which the SDK picks up as a fallback
        # even when we explicitly pass base_url=None.
        effective_base_url = base_url if base_url else None
        if not os.environ.get("OPENAI_BASE_URL"):
            os.environ.pop("OPENAI_BASE_URL", None)
        self._client = client or AsyncOpenAI(
            api_key=api_key,
            base_url=effective_base_url,
        )
        self._lc_agent = build_langchain_agent(model=model, api_key=api_key)

    # Keywords that suggest a calculator tool call.
    _CALC_KW = (
        "calcul",
        "combien font",
        "combien fait",
        "addition",
        "soustraction",
        "multipli",
        "divis",
        "puissance",
        "exposant",
        "racine",
        "modulo",
        "factori",
    )

    # Keywords that suggest a web-search tool call.
    _WEB_KW = (
        "recherche sur",
        "cherche sur",
        "fais une recherche",
        "dis m'en plus sur",
        "dis-moi plus sur",
        "dis moi plus sur",
        "search the web",
        "web search",
        "cherche sur le web",
        "cherche sur internet",
        "recherche web",
        "en savoir plus sur",
        "infos sur",
        "informations sur",
    )

    # Keywords that suggest a draft-reply request.
    _DRAFT_KW = (
        "brouillon",
        "draft",
        "rédig",
        "redig",
        "génère une réponse",
        "genere une reponse",
        "génères une réponse",
        "generes une reponse",
        "que répondre",
        "que repondre",
        "comment répondre",
        "comment repondre",
        "génère un mail",
        "genere un mail",
        "écris un mail",
        "ecris un mail",
        "réponds à ce mail",
        "reponds a ce mail",
    )

    # -- App-action keywords ------------------------------------------------
    # Each tuple entry is either a plain substring match OR a frozenset
    # of words that must ALL appear in the message (bag-of-words match).
    _GMAIL_CONNECT_KW = (
        "connecte gmail",
        "connecter gmail",
        "connexion gmail",
        "connect gmail",
        "login gmail",
        "se connecter à gmail",
        "se connecter a gmail",
        "connecter à gmail",
        "connecter a gmail",
        "connecte-moi",
        frozenset({"connecte", "gmail"}),
        frozenset({"connecter", "gmail"}),
    )
    _GMAIL_SYNC_KW = (
        "synchronise",
        "synchro",
        "sync",
        "récupère mes mails",
        "recupere mes mails",
        "récupérer mes mails",
        "recuperer mes mails",
        "charge mes mails",
        "importe mes mails",
        "télécharge mes mails",
        "telecharge mes mails",
        "fetch mails",
        "fetch emails",
    )
    _CLASSIFY_MAILS_KW = (
        "classe mes mails",
        "classifier mes mails",
        "classifie mes mails",
        "catégorise mes mails",
        "categorise mes mails",
        "trie mes mails",
        "classify mails",
        "classify emails",
        frozenset({"classe", "mails"}),
        frozenset({"trie", "mails"}),
    )
    _CLASSIFY_DOCS_KW = (
        "classe mes documents",
        "classifier mes documents",
        "classifie mes documents",
        "catégorise mes documents",
        "categorise mes documents",
        "trie mes documents",
        "classe mes fichiers",
        "classify documents",
        "classify docs",
        frozenset({"classe", "documents"}),
        frozenset({"classe", "fichiers"}),
        frozenset({"trie", "documents"}),
    )
    _GMAIL_STATUS_KW = (
        "gmail connecté",
        "gmail connecte",
        "gmail est connecté",
        "gmail est connecte",
        "statut gmail",
        "status gmail",
        "état de gmail",
        "etat de gmail",
        "gmail status",
        "gmail connected",
        frozenset({"statut", "gmail"}),
        frozenset({"status", "gmail"}),
        frozenset({"gmail", "est", "connecte"}),
        frozenset({"gmail", "est", "connecté"}),
    )
    _LIST_MAILS_KW = (
        "liste mes mails",
        "lister mes mails",
        "montre mes mails",
        "affiche mes mails",
        "mes derniers mails",
        "list mails",
        "list emails",
        "show my mails",
        frozenset({"liste", "mails"}),
        frozenset({"montre", "mails"}),
        frozenset({"affiche", "mails"}),
    )

    @staticmethod
    def _match(keywords: tuple[str | frozenset[str], ...], text: str) -> bool:
        """Check if any keyword matches: substring for str, all-words for frozenset."""
        # Split on whitespace AND hyphens so "connecte-moi" yields {"connecte", "moi"}
        words: set[str] = set()
        for w in text.split():
            words.update(w.split("-"))
        for k in keywords:
            if isinstance(k, frozenset):
                if k <= words:
                    return True
            elif k in text:
                return True
        return False

    def _needs_tool(self, message: str) -> str | None:
        """Return the tool category or None."""
        low = message.casefold()
        if any(k in low for k in self._CALC_KW):
            return "calc"
        if any(k in low for k in self._WEB_KW):
            return "web"
        if any(k in low for k in self._DRAFT_KW):
            return "draft"
        # App actions — check status BEFORE connect to avoid
        # "gmail est connecte" matching the connect action.
        if self._match(self._GMAIL_STATUS_KW, low):
            return "gmail_status"
        if self._match(self._GMAIL_CONNECT_KW, low):
            return "gmail_connect"
        if self._match(self._CLASSIFY_MAILS_KW, low):
            return "classify_mails"
        if self._match(self._CLASSIFY_DOCS_KW, low):
            return "classify_docs"
        if self._match(self._LIST_MAILS_KW, low):
            return "list_mails"
        if self._match(self._GMAIL_SYNC_KW, low):
            return "gmail_sync"
        return None

    async def chat(
        self,
        *,
        message: str,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, object]:
        tool = self._needs_tool(message)
        logger.info(
            "Tool routing: %s | message=%s",
            tool or "RAG (no tool match)",
            message[:80],
        )

        # --- Draft reply: use the dedicated tool with mail context ---
        if tool == "draft":
            return await self._handle_draft(message, history)

        # --- App actions (Gmail connect, sync, classify, status, list) ---
        app_tools = (
            "gmail_connect",
            "gmail_sync",
            "classify_mails",
            "classify_docs",
            "gmail_status",
            "list_mails",
        )
        if tool in app_tools:
            return await self._handle_app_action(tool)

        # --- Calculator / Web search: route to LangChain agent ---
        if tool in ("calc", "web"):
            try:
                result = self._lc_agent.invoke(
                    {"messages": [{"role": "user", "content": message}]}
                )
                final_message = result["messages"][-1].content
                logger.info(
                    "LangChain result: %s",
                    (final_message or "")[:200],
                )

                if isinstance(final_message, str) and final_message.strip():
                    return {
                        "answer": final_message.strip(),
                        "sources": [],
                    }
            except Exception as exc:
                logger.warning("LangChain agent failed: %s", exc)

        # --- RAG principal (mails, documents, etc.) ---
        route = self._route(message)
        sources = self._search(message, route)
        temporal = self._temporal_context(message, route, history)
        answer = await self._answer(message, sources, route, history, temporal)

        return {
            "answer": answer,
            "sources": [s.to_dict() for s in sources],
        }

    async def classify_email(self, *, email_text: str) -> dict[str, str]:
        return await _classify_email(self._client, self._model, email_text)

    async def draft_email_reply(self, *, email_text: str) -> str:
        return await _draft_email_reply(self._client, self._model, email_text)

    async def _handle_draft(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, object]:
        """Build email context from history/temporal data and call draft tool."""
        # Try to recover the mail body from recent conversation or temporal DB
        email_text = self._extract_mail_context(message, history)
        if email_text:
            logger.info("Drafting reply via draft_email_reply tool")
            try:
                draft = await self.draft_email_reply(
                    email_text=email_text,
                )
                return {"answer": draft, "sources": []}
            except Exception as exc:
                logger.warning("draft_email_reply failed: %s", exc)
        # Fall back to RAG-based answer if no mail context found
        logger.info("No mail context for draft, falling back to RAG")
        route = self._route(message)
        sources = self._search(message, route)
        temporal = self._temporal_context(message, route, history)
        answer = await self._answer(
            message, sources, route, history, temporal,
        )
        return {
            "answer": answer,
            "sources": [s.to_dict() for s in sources],
        }

    def _extract_mail_context(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
    ) -> str | None:
        """Try to find mail content from conversation history or DB."""
        # 1. Look in recent assistant messages for a mail body
        if history:
            for h in reversed(history[-6:]):
                if h.get("role") != "assistant":
                    continue
                text = h.get("content", "")
                # Heuristic: if the assistant quoted a mail subject/body
                if (
                    len(text) > 100
                    and ("sujet" in text.casefold()
                         or "mail" in text.casefold()
                         or "@" in text)
                ):
                    return text
        # 2. Try the latest received mail from the DB
        if self._store:
            mails = self._store.search(
                doc_type="mail", direction="received", limit=1,
            )
            if mails:
                fp = str(mails[0].get("file_path", ""))
                if fp:
                    body = self._kb.read_chunk(
                        relative_path=fp, chunk_index=None,
                    )
                    if body and body.strip():
                        return body[:3000].strip()
        return None

    # ------------------------------------------------------------------
    # App-action handlers
    # ------------------------------------------------------------------

    async def _handle_app_action(self, action: str) -> dict[str, object]:
        """Execute an app-level action and return a chat-style response."""
        handler = {
            "gmail_connect": self._action_gmail_connect,
            "gmail_sync": self._action_gmail_sync,
            "classify_mails": self._action_classify_mails,
            "classify_docs": self._action_classify_docs,
            "gmail_status": self._action_gmail_status,
            "list_mails": self._action_list_mails,
        }.get(action)
        if not handler:
            return self._app_reply(
                "Action inconnue. Essaie : "
                "synchronise, classe mes mails, statut gmail…"
            )
        logger.info("Executing app action: %s", action)
        try:
            return await handler()
        except Exception as exc:
            logger.error("App action %s failed: %s", action, exc)
            return self._app_reply(f"❌ Erreur : {exc}")

    @staticmethod
    def _app_reply(text: str) -> dict[str, object]:
        return {"answer": text, "sources": []}

    async def _action_gmail_connect(self) -> dict[str, object]:
        if not self._gmail:
            return self._app_reply(
                "⚠️ Le handler Gmail n'est pas configuré."
            )
        if self._gmail.is_connected():
            return self._app_reply(
                "✅ Gmail est déjà connecté."
            )
        ok = self._gmail.connect()
        if ok:
            return self._app_reply(
                "✅ Gmail connecté avec succès ! "
                "Tu peux maintenant synchroniser tes mails."
            )
        return self._app_reply(
            "❌ Échec de la connexion Gmail. "
            "Vérifie les identifiants OAuth."
        )

    async def _action_gmail_sync(self) -> dict[str, object]:
        if not self._gmail:
            return self._app_reply(
                "⚠️ Le handler Gmail n'est pas configuré."
            )
        if not self._gmail.is_connected():
            return self._app_reply(
                "⚠️ Gmail n'est pas connecté. "
                "Dis-moi « connecte Gmail » d'abord."
            )
        result = await self._gmail.sync()
        synced = result.get("synced", 0)
        indexed = result.get("indexed", 0)
        parts = [
            "✅ Synchronisation terminée !",
            f"- **{result.get('received', 0)}** mails reçus récupérés",
            f"- **{result.get('sent', 0)}** mails envoyés récupérés",
            f"- **{synced}** nouveaux mails sauvegardés",
            f"- **{indexed}** documents indexés",
        ]
        # Auto-classify after sync
        classified = await self.classify_uncategorized_mails()
        if classified:
            parts.append(
                f"- **{classified}** mails classifiés automatiquement"
            )
        return self._app_reply("\n".join(parts))

    async def _action_classify_mails(self) -> dict[str, object]:
        count = await self.classify_uncategorized_mails()
        if count:
            return self._app_reply(
                f"🏷️ **{count}** mails classifiés avec succès."
            )
        return self._app_reply(
            "✅ Tous les mails sont déjà classifiés."
        )

    async def _action_classify_docs(self) -> dict[str, object]:
        count = await self.classify_uncategorized_docs()
        if count:
            return self._app_reply(
                f"🏷️ **{count}** documents classifiés avec succès."
            )
        return self._app_reply(
            "✅ Tous les documents sont déjà classifiés."
        )

    async def _action_gmail_status(self) -> dict[str, object]:
        if not self._gmail:
            return self._app_reply(
                "⚠️ Le handler Gmail n'est pas configuré."
            )
        connected = self._gmail.is_connected()
        status = "connecté ✅" if connected else "déconnecté ❌"
        parts = [f"**Gmail** : {status}"]
        if connected:
            age = self._gmail.sync_age_hours()
            if age is not None:
                if age < 1:
                    parts.append(
                        "Dernière synchro : il y a moins d'une heure"
                    )
                else:
                    parts.append(
                        f"Dernière synchro : "
                        f"il y a **{age:.0f}** heures"
                    )
            else:
                parts.append("Aucune synchronisation effectuée.")
        if self._store:
            total = self._store.count(doc_type="mail")
            uncat = len(self._store.uncategorized_mails())
            parts.append(f"**{total}** mails indexés")
            if uncat:
                parts.append(f"**{uncat}** mails non classifiés")
        return self._app_reply("\n".join(parts))

    async def _action_list_mails(self) -> dict[str, object]:
        if not self._store:
            return self._app_reply(
                "Aucun mail indexé pour le moment."
            )
        mails = self._store.search(doc_type="mail", limit=10)
        if not mails:
            return self._app_reply(
                "Aucun mail trouvé. "
                "Synchronise Gmail d'abord."
            )
        lines = ["📬 **Derniers mails** :\n"]
        for m in mails:
            d = str(m.get("direction", ""))
            icon = "📥" if d == "received" else "📤"
            subj = str(m.get("title", "Sans sujet"))
            cat = str(m.get("category", "")) or "—"
            date = str(m.get("date", ""))[:16]
            person = (
                str(m.get("sender", ""))
                if d == "received"
                else str(m.get("recipient", ""))
            )
            lines.append(
                f"{icon} **{subj}**\n"
                f"   {person} · {date} · {cat}"
            )
        return self._app_reply("\n".join(lines))

    async def _classify_with_retry(
        self, doc_id: int, title: str, *, retries: int = 2
    ) -> bool:
        """Try to classify a single document, retrying on transient errors."""
        for attempt in range(1, retries + 1):
            try:
                await _classify_and_store(
                    self._client,
                    self._model,
                    self._store,  # type: ignore[arg-type]
                    doc_id,
                    title,
                )
                return True
            except Exception as exc:
                if attempt < retries:
                    logger.warning(
                        "Classification attempt %d/%d failed for doc %d (%s): %s",
                        attempt, retries, doc_id, title, exc,
                    )
                    await asyncio.sleep(1.0 * attempt)
                else:
                    logger.error(
                        "Classification failed for doc %d (%s) after %d attempts: %s",
                        doc_id, title, retries, exc,
                    )
        return False

    async def classify_uncategorized_mails(self) -> int:
        """Classify all mails without a category. Returns count classified."""
        if not self._store:
            return 0
        mails = self._store.uncategorized_mails()
        classified = 0
        failed = 0
        for mail in mails:
            ok = await self._classify_with_retry(
                int(mail["id"]), str(mail["title"])
            )
            if ok:
                classified += 1
            else:
                failed += 1
        if failed:
            logger.warning(
                "Mail classification: %d succeeded, %d failed out of %d",
                classified, failed, len(mails),
            )
        return classified

    async def classify_uncategorized_docs(self) -> int:
        """Classify all uploaded docs without a category."""
        if not self._store:
            return 0
        docs = self._store.uncategorized_docs()
        classified = 0
        failed = 0
        for doc in docs:
            ok = await self._classify_with_retry(
                int(doc["id"]), str(doc["title"])
            )
            if ok:
                classified += 1
            else:
                failed += 1
        if failed:
            logger.warning(
                "Doc classification: %d succeeded, %d failed out of %d",
                classified, failed, len(docs),
            )
        return classified

    # ----------------------------------------------------------------------
    # Routing Gmail / general
    # ----------------------------------------------------------------------

    _DOC_KW = (
        " cv ",
        " mon cv ",
        " resume ",
        " mon profil ",
        " profil ",
        " diplôme ",
        " diplome ",
        " diplômes ",
        " diplomes ",
        " formation ",
        " mes formations ",
        # Use unpadded substrings for words commonly preceded by
        # French elision (d', l', etc.) so they match "d'expérience"
        "compétence",
        "competence",
        "expérience",
        "experience",
        " parcours ",
        " mon parcours ",
        " carrière ",
        " carriere ",
        " ma carrière ",
        " ma carriere ",
        " quel poste ",
        " quels postes ",
        " attestation ",
        " ects ",
        " cursus ",
        "études",
        "etudes",
        " école ",
        " ecole ",
        "l'école",
        "l'ecole",
        " portfolio ",
        " mes projets ",
        " mon parcours pro ",
        " pour quel poste ",
        " quel job ",
        " quel emploi ",
        " quels emplois ",
        " mes skills ",
        " skills ",
        " mes documents ",
        " document uploadé ",
        " document uploade ",
        " fichier uploadé ",
        " fichier uploade ",
    )

    # Prefix used to exclude Gmail paths when routing to documents only.
    _GMAIL_PREFIX = "from_gmail/"

    def _route(self, message: str) -> dict[str, object]:
        n = f" {message.casefold()} "

        sent_kw = (
            " sent ",
            " envoye ",
            " envoyé ",
            " envoyés ",
            " envoyes ",
            " ai-je envoye ",
            " ai je envoye ",
            " ai-je envoyé ",
            " ai je envoyé ",
            " j ai envoye ",
            " j ai envoyé ",
            " mail envoyé ",
            " mails envoyés ",
            " email envoyé ",
            " a qui ai-je ecrit ",
            " a qui ai je ecrit ",
            " a qui ai-je écrit ",
            " a qui ai je écrit ",
            " ai-je repondu ",
            " ai je repondu ",
            " ai-je répondu ",
            " ai je répondu ",
        )

        recv_kw = (
            " received ",
            " recu ",
            " reçu ",
            " recus ",
            " reçus ",
            " ai-je recu ",
            " ai je recu ",
            " ai-je reçu ",
            " ai je reçu ",
            " on m'a envoyé ",
            " on m a envoye ",
            " on m a envoyé ",
            " boite de reception ",
            " boite de réception ",
            " inbox ",
            " qui m'a ecrit ",
            " qui m a ecrit ",
            " qui m'a écrit ",
            " qui m a écrit ",
        )

        if any(k in n for k in sent_kw):
            return {
                "label": "sent",
                "prefixes": ["from_gmail/sent/"],
                "strict": True,
            }

        if any(k in n for k in recv_kw):
            return {
                "label": "received",
                "prefixes": ["from_gmail/received/"],
                "strict": False,
            }

        # Document / profile queries → exclude mails, search only uploads
        if any(k in n for k in self._DOC_KW):
            return {
                "label": "docs",
                "prefixes": None,
                "exclude_prefixes": [self._GMAIL_PREFIX],
                "strict": False,
            }

        return {"label": "all", "prefixes": None, "strict": False}

    # -- temporal ------------------------------------------------------------

    _TEMPORAL_KW = (
        " dernier ",
        " derniers ",
        " dernière ",
        " dernieres ",
        " plus récent ",
        " plus recent ",
        " récemment ",
        " recemment ",
        " aujourd'hui ",
        " aujourd hui ",
        " ce matin ",
        " ce soir ",
        " cette semaine ",
        " hier ",
        " avant-hier ",
        " tout à l'heure ",
        " tout a l heure ",
    )

    _REFERENCE_KW = (
        " ce mail ",
        " cet email ",
        " cet e-mail ",
        " ce message ",
        " ce dernier ",
        " son contenu ",
        " le contenu ",
        " le résumer ",
        " le resumer ",
        " en parle ",
        " de quoi parle ",
        " qu'est-ce qu'il dit ",
        " qu est-ce qu il dit ",
        " il dit quoi ",
        " ça parle de quoi ",
        " ca parle de quoi ",
    )

    def _needs_temporal(
        self,
        message: str,
        history: list[dict[str, str]] | None,
    ) -> bool:
        """Detect temporal queries or follow-up references to a mail."""
        n = f" {message.casefold()} "
        if any(k in n for k in self._TEMPORAL_KW):
            return True
        if any(k in n for k in self._REFERENCE_KW) and history:
            recent = [
                h["content"]
                for h in (history or [])[-4:]
                if h.get("role") == "assistant"
            ]
            text = " ".join(recent).casefold()
            return "mail" in text or "sujet" in text or "@" in text
        return False

    def _temporal_context(
        self,
        message: str,
        route: dict[str, object],
        history: list[dict[str, str]] | None = None,
    ) -> str | None:
        """If the query is temporal or a follow-up, return recent mails with content."""
        if not self._store:
            return None
        if not self._needs_temporal(message, history):
            return None
        direction = None
        label = route.get("label")
        if label == "sent":
            direction = "sent"
        elif label == "received":
            direction = "received"
        mails = self._store.search(doc_type="mail", direction=direction, limit=5)
        if not mails:
            return None
        lines: list[str] = ["Mails les plus récents (triés par date) :"]
        for m in mails:
            d = str(m.get("direction", ""))
            subj = str(m.get("title", ""))
            sender = str(m.get("sender", ""))
            recip = str(m.get("recipient", ""))
            date = str(m.get("date", ""))
            cat = str(m.get("category", "")) or "—"
            person = sender if d == "received" else recip
            fp = str(m.get("file_path", ""))
            content = (
                self._kb.read_chunk(relative_path=fp, chunk_index=None)[:2000].strip()
                if fp
                else ""
            )
            lines.append(
                f"- [{d}] {date} | De/À: {person} | Sujet: {subj} | Cat: {cat}"
            )
            if content:
                lines.append(f"  Contenu:\n{content}")
        return "\n".join(lines)

    # -- search --------------------------------------------------------------

    def _search(self, message: str, route: dict[str, object]) -> list[DocumentMatch]:
        raw_prefixes = route.get("prefixes")
        prefixes = raw_prefixes if isinstance(raw_prefixes, list) else None
        raw_exclude = route.get("exclude_prefixes")
        exclude_prefixes = raw_exclude if isinstance(raw_exclude, list) else None

        if prefixes:
            scoped = self._kb.search(
                query=message,
                limit=self._search_limit,
                prefixes=prefixes,
            )

            if route.get("strict") or len(scoped) >= min(2, self._search_limit):
                return scoped

            fallback = self._kb.search(
                query=message,
                limit=self._search_limit,
            )

            return self._merge(scoped, fallback)

        if exclude_prefixes:
            scoped = self._kb.search(
                query=message,
                limit=self._search_limit,
                exclude_prefixes=exclude_prefixes,
            )
            # If we found enough doc results, return them;
            # otherwise merge with a full search for broader context.
            if len(scoped) >= min(2, self._search_limit):
                return scoped

            fallback = self._kb.search(
                query=message,
                limit=self._search_limit,
            )
            return self._merge(scoped, fallback)

        return self._kb.search(query=message, limit=self._search_limit)

    def _merge(
        self, a: list[DocumentMatch], b: list[DocumentMatch]
    ) -> list[DocumentMatch]:
        merged: list[DocumentMatch] = []
        seen: set[tuple[str, int | None]] = set()

        for s in [*a, *b]:
            key = (s.relative_path, s.chunk_index)

            if key in seen:
                continue

            seen.add(key)
            merged.append(s)

            if len(merged) >= self._search_limit:
                break

        return merged

    # ----------------------------------------------------------------------
    # LLM
    # ----------------------------------------------------------------------

    async def _answer(
        self,
        message: str,
        sources: list[DocumentMatch],
        route: dict[str, object],
        history: list[dict[str, str]] | None = None,
        temporal: str | None = None,
    ) -> str:
        ctx = self._build_context(sources, route)
        conv = self._format_history(history)
        parts = [
            f"Historique de conversation:\n{conv}",
            f"Question utilisateur:\n{message}",
        ]
        if temporal:
            parts.append(f"Données temporelles (SQLite):\n{temporal}")
        parts.append(f"Contexte RAG:\n{ctx}")
        user_input = "\n\n".join(parts)

        resp = await self._client.responses.create(
            model=self._model,
            instructions=(
                "Tu es Resumi, un assistant personnel organisé. "
                "Réponds en français, de manière concise et claire. "
                "Utilise l'historique de conversation pour comprendre "
                "le contexte de la question. "
                "Pour les questions temporelles (dernier, récent, "
                "aujourd'hui…), utilise en priorité les données "
                "temporelles SQLite qui sont triées par date. "
                "Utilise aussi le contexte RAG fourni. "
                "Quand le contexte indique sent ou received, "
                "respecte cette portée. "
                "Quand le contexte indique docs, les sources sont les "
                "documents personnels de l'utilisateur (CV, diplômes, "
                "attestations) — base-toi dessus pour répondre aux "
                "questions sur son profil, son parcours et ses compétences. "
                "Si le contexte ne suffit pas, dis-le "
                "explicitement sans inventer."
            ),
            input=user_input,
        )

        text = resp.output_text.strip()

        if text:
            return text

        return (
            "Je n'ai pas pu générer de réponse. "
            "Réessaie après avoir indexé plus de documents."
        )

    def _build_context(
        self,
        sources: list[DocumentMatch],
        route: dict[str, object],
    ) -> str:
        if not sources:
            return (
                f"Routage: {route['label']}\n"
                "Aucun document pertinent n'a encore été indexé. "
                "Invite l'utilisateur à envoyer un audio ou lancer la sync Gmail."
            )

        parts = [f"Routage: {route['label']}"]

        for s in sources:
            chunk = self._kb.read_chunk(
                relative_path=s.relative_path,
                chunk_index=s.chunk_index,
            )

            snippet = (s.excerpt or chunk)[:1800].strip()

            parts.append(
                f"Source: {s.relative_path}\n"
                f"Boîte: {s.mailbox or 'all'}\n"
                f"Chunk: {s.chunk_index}\n"
                f"Score: {s.score:.3f}\n"
                f"Contenu:\n{snippet}"
            )
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _format_history(
        history: list[dict[str, str]] | None,
        max_turns: int = 10,
    ) -> str:
        """Format recent conversation history for the LLM."""
        if not history:
            return "(aucun historique)"
        turns = [h for h in history if h.get("role") in ("user", "assistant")]
        recent = turns[-max_turns * 2 :]
        lines: list[str] = []
        for h in recent:
            role = "Utilisateur" if h["role"] == "user" else "Assistant"
            lines.append(f"{role}: {h['content'][:500]}")
        return "\n".join(lines) if lines else "(aucun historique)"