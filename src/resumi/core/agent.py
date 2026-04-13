"""RAG-powered assistant agent."""

from openai import AsyncOpenAI

from resumi.core.embedding import DocumentMatch, FaissKnowledgeBase
from resumi.core.web_search import web_search


class Agent:
    def __init__(
        self,
        *,
        knowledge_base: FaissKnowledgeBase,
        model: str,
        api_key: str,
        base_url: str | None = None,
        search_limit: int = 8,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self._kb = knowledge_base
        self._model = model
        self._search_limit = search_limit
        self._client = client or AsyncOpenAI(
            api_key=api_key,
            base_url=base_url or None,
        )

    async def chat(self, *, message: str) -> dict[str, object]:
        # --- Web search routing ---
        if self._should_use_web_search(message):
            raw = web_search(message)

            resp = await self._client.responses.create(
                model=self._model,
                instructions=(
                    "Tu es un assistant utile et concis. "
                    "Améliore, reformule et clarifie la réponse suivante de manière "
                    "naturelle et fiable en français. "
                    "Si la réponse est limitée ou incomplète, dis-le clairement."
                ),
                input=raw,
            )

            return {
                "answer": resp.output_text.strip(),
                "sources": [],
            }

        # --- RAG pipeline ---
        route = self._route(message)
        sources = self._search(message, route)
        answer = await self._answer(message, sources, route)

        return {
            "answer": answer,
            "sources": [s.to_dict() for s in sources],
        }

    async def classify_email(self, *, email_text: str) -> dict[str, str]:
        resp = await self._client.responses.create(
            model=self._model,
            instructions=(
                "Tu es un assistant de classification d'emails. "
                "Tu dois classer l'email dans UNE seule catégorie parmi : "
                "Personnel, Professionnel, Administratif, Académique, Commercial, Autre. "
                "Tu dois aussi proposer un niveau de priorité parmi : basse, moyenne, haute. "
                "Réponds uniquement en JSON valide avec les clés : "
                "category, priority, reason."
            ),
            input=f"Email à analyser :\n\n{email_text}",
        )

        text = resp.output_text.strip()
        return {"raw_result": text}

    async def draft_email_reply(self, *, email_text: str) -> str:
        resp = await self._client.responses.create(
            model=self._model,
            instructions=(
                "Tu es un assistant de rédaction d'emails. "
                "Rédige un brouillon de réponse en français, clair, poli, concis et naturel. "
                "N'invente pas d'informations précises qui ne figurent pas dans l'email. "
                "Le ton doit être professionnel mais humain. "
                "Retourne uniquement le texte du brouillon."
            ),
            input=(
                f"Voici l'email reçu :\n\n{email_text}\n\n"
                "Rédige une réponse adaptée."
            ),
        )

        text = resp.output_text.strip()

        if text:
            return text

        return "Impossible de générer un brouillon pour le moment."

    # ----------------------------------------------------------------------
    # Web search routing
    # ----------------------------------------------------------------------

    def _should_use_web_search(self, message: str) -> bool:
        n = message.casefold()

        web_keywords = (
            "recherche web",
            "cherche sur le web",
            "cherche sur internet",
            "sur internet",
            "sur le web",
            "web search",
            "search the web",
            "search online",
            "internet",
            "actualité",
            "news",
            "météo",
            "weather",
        )

        return any(k in n for k in web_keywords)

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

    # ----------------------------------------------------------------------
    # Search
    # ----------------------------------------------------------------------

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
    ) -> str:
        ctx = self._build_context(sources, route)

        resp = await self._client.responses.create(
            model=self._model,
            instructions=(
                "Tu es Resumi, un assistant personnel organisé. "
                "Réponds en français, de manière concise et claire. "
                "Utilise en priorité le contexte fourni. "
                "Quand le contexte indique sent ou received, respecte cette portée. "
                "Si le contexte ne suffit pas, dis-le explicitement sans inventer."
            ),
            input=f"Question utilisateur:\n{message}\n\nContexte RAG:\n{ctx}",
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
    