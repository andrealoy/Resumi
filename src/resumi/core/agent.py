"""RAG-powered assistant agent."""

from openai import AsyncOpenAI

from resumi.core.embedding import DocumentMatch, FaissKnowledgeBase


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
        self._client = client or AsyncOpenAI(api_key=api_key, base_url=base_url or None)

    async def chat(self, *, message: str) -> dict[str, object]:
        route = self._route(message)
        sources = self._search(message, route)
        answer = await self._answer(message, sources, route)
        return {
            "answer": answer,
            "sources": [s.to_dict() for s in sources],
        }

    # -- routing -------------------------------------------------------------

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
            return {"label": "sent", "prefixes": ["from_gmail/sent/"], "strict": True}
        if any(k in n for k in recv_kw):
            return {
                "label": "received",
                "prefixes": ["from_gmail/received/"],
                "strict": False,
            }
        return {"label": "all", "prefixes": None, "strict": False}

    # -- search --------------------------------------------------------------

    def _search(self, message: str, route: dict[str, object]) -> list[DocumentMatch]:
        prefixes = route.get("prefixes")  # type: ignore[arg-type]
        if prefixes:
            scoped = self._kb.search(
                query=message, limit=self._search_limit, prefixes=prefixes
            )
            if route.get("strict") or len(scoped) >= min(2, self._search_limit):
                return scoped
            fallback = self._kb.search(query=message, limit=self._search_limit)
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

    # -- LLM -----------------------------------------------------------------

    async def _answer(
        self, message: str, sources: list[DocumentMatch], route: dict[str, object]
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
        self, sources: list[DocumentMatch], route: dict[str, object]
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
                relative_path=s.relative_path, chunk_index=s.chunk_index
            )
            snippet = (s.excerpt or chunk)[:1800].strip()
            parts.append(
                f"Source: {s.relative_path}\nBoîte: {s.mailbox or 'all'}\n"
                f"Chunk: {s.chunk_index}\nScore: {s.score:.3f}\nContenu:\n{snippet}"
            )
        return "\n\n---\n\n".join(parts)
