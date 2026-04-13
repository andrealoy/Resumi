"""RAG-powered assistant agent."""

from openai import AsyncOpenAI

from resumi.core.document_store import DocumentStore
from resumi.core.embedding import DocumentMatch, FaissKnowledgeBase
from resumi.core.langchain_agent import build_langchain_agent
from resumi.core.mail_tools import classify_and_store as _classify_and_store
from resumi.core.mail_tools import classify_email as _classify_email
from resumi.core.mail_tools import draft_email_reply as _draft_email_reply


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
    ) -> None:
        self._kb = knowledge_base
        self._model = model
        self._search_limit = search_limit
        self._store = store
        self._client = client or AsyncOpenAI(
            api_key=api_key,
            base_url=base_url or None,
        )
        self._lc_agent = build_langchain_agent(model=model, api_key=api_key)

    async def chat(
        self,
        *,
        message: str,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, object]:
        # --- Agent LangChain (tool calling) ---
        try:
            result = self._lc_agent.invoke(
                {"messages": [{"role": "user", "content": message}]}
            )
            final_message = result["messages"][-1].content

            if isinstance(final_message, str) and final_message.strip():
                return {
                    "answer": final_message.strip(),
                    "sources": [],
                }
        except Exception:
            pass

        # --- Fallback RAG actuel ---
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

    async def classify_uncategorized_mails(self) -> int:
        """Classify all mails without a category. Returns count classified."""
        if not self._store:
            return 0
        mails = self._store.uncategorized_mails()
        classified = 0
        for mail in mails:
            try:
                await _classify_and_store(
                    self._client,
                    self._model,
                    self._store,
                    int(mail["id"]),
                    str(mail["title"]),
                )
                classified += 1
            except Exception:
                continue
        return classified

    async def classify_uncategorized_docs(self) -> int:
        """Classify all uploaded docs without a category."""
        if not self._store:
            return 0
        docs = self._store.uncategorized_docs()
        classified = 0
        for doc in docs:
            try:
                await _classify_and_store(
                    self._client,
                    self._model,
                    self._store,
                    int(doc["id"]),
                    str(doc["title"]),
                )
                classified += 1
            except Exception:
                continue
        return classified

    # ----------------------------------------------------------------------
    # Routing Gmail / general
    # ----------------------------------------------------------------------

    def _route(self, message: str) -> dict[str, object]:
        n = f" {message.casefold()} "

        sent_kw = (
            " sent ",
            " envoye ",
            " envoyés ",
            " envoyes ",
            " ai-je envoye ",
            " ai je envoye ",
            " j ai envoye ",
            " mail envoyé ",
            " mails envoyés ",
            " email envoyé ",
            " a qui ai-je ecrit ",
            " a qui ai je ecrit ",
            " ai-je repondu ",
            " ai je repondu ",
        )

        recv_kw = (
            " received ",
            " recu ",
            " recus ",
            " reçus ",
            " ai-je recu ",
            " ai je recu ",
            " on m'a envoyé ",
            " on m a envoye ",
            " boite de reception ",
            " inbox ",
            " qui m'a ecrit ",
            " qui m a ecrit ",
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