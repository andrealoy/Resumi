"""Embedding & FAISS vector search."""

import json
from pathlib import Path
from typing import Any, Protocol, TypedDict, cast

import faiss  # type: ignore[import-untyped]
import numpy as np
import pymupdf
from openai import OpenAI

# ---------------------------------------------------------------------------
# Embedder protocol + implementations
# ---------------------------------------------------------------------------


class TextEmbedder(Protocol):
    @property
    def dimension(self) -> int: ...

    def embed_texts(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, query: str) -> np.ndarray: ...


class LocalHashEmbedder:
    """Deterministic hash-based embedder (fallback when no API key)."""

    def __init__(self, *, dimension: int = 128) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dimension), dtype="float32")
        return np.vstack([self._embed(t) for t in texts])

    def embed_query(self, query: str) -> np.ndarray:
        return self._embed(query)

    def _embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dimension, dtype="float32")
        for token in _normalize_tokens(text):
            vec[hash(token) % self._dimension] += 1.0
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        return vec


class OpenAITextEmbedder:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str | None,
        fallback: TextEmbedder,
        dimensions: int | None = None,
        batch_size: int = 32,
    ) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url or None)
        self._model = model
        self._dimensions = dimensions
        self._batch_size = batch_size
        self._fallback = fallback

    @property
    def dimension(self) -> int:
        return (
            self._dimensions
            if self._dimensions is not None
            else self._fallback.dimension
        )

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype="float32")
        batches: list[np.ndarray] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            try:
                batches.append(self._call_api(batch))
            except Exception:
                batches.append(self._fallback.embed_texts(batch))
        return np.vstack(batches).astype("float32")

    def embed_query(self, query: str) -> np.ndarray:
        try:
            return np.asarray(self._call_api([query])[0], dtype="float32")
        except Exception:
            return self._fallback.embed_query(query)

    def _call_api(self, texts: list[str]) -> np.ndarray:
        api = cast(Any, self._client.embeddings)
        kwargs: dict[str, Any] = {"model": self._model, "input": texts}
        if self._dimensions is not None:
            kwargs["dimensions"] = self._dimensions
        resp = api.create(**kwargs)
        return np.array([item.embedding for item in resp.data], dtype="float32")


# ---------------------------------------------------------------------------
# FAISS knowledge base
# ---------------------------------------------------------------------------

QUERY_SYNONYMS: dict[str, tuple[str, ...]] = {
    "coiffeur": ("coiffure", "salon", "barber", "hair"),
    "coiffeuse": ("coiffure", "salon", "barber", "hair"),
    "rendez": ("appointment", "booking", "reservation"),
    "rdv": ("appointment", "booking", "reservation"),
    "dentiste": ("dentist", "cabinet", "appointment"),
    "docteur": ("doctor", "medecin", "consultation"),
    "médecin": ("doctor", "medecin", "consultation"),
}


class ChunkMeta(TypedDict):
    title: str
    relative_path: str
    chunk_index: int
    chunk_text: str


class DocumentMatch:
    __slots__ = ("title", "relative_path", "score", "chunk_index", "excerpt", "mailbox")

    def __init__(
        self,
        *,
        title: str,
        relative_path: str,
        score: float,
        chunk_index: int | None = None,
        excerpt: str = "",
        mailbox: str | None = None,
    ) -> None:
        self.title = title
        self.relative_path = relative_path
        self.score = score
        self.chunk_index = chunk_index
        self.excerpt = excerpt
        self.mailbox = mailbox

    def to_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "relative_path": self.relative_path,
            "score": self.score,
            "chunk_index": self.chunk_index,
            "excerpt": self.excerpt,
            "mailbox": self.mailbox,
        }


class FaissKnowledgeBase:
    def __init__(
        self,
        *,
        docs_root: str,
        index_root: str,
        embedder: TextEmbedder | None = None,
        chunk_size: int = 160,
        chunk_overlap: int = 40,
        candidate_pool: int = 24,
    ) -> None:
        self._docs = Path(docs_root)
        self._index_root = Path(index_root)
        self._embedder = embedder or LocalHashEmbedder(dimension=128)
        self._dim = self._embedder.dimension
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._candidate_pool = candidate_pool
        self._index_path = self._index_root / "documents.faiss"
        self._meta_path = self._index_root / "documents.json"
        self._index_root.mkdir(parents=True, exist_ok=True)

    # -- indexing ------------------------------------------------------------

    _TEXT_EXTS = {".md", ".txt"}
    _PDF_EXTS = {".pdf"}
    _ALL_EXTS = _TEXT_EXTS | _PDF_EXTS

    @staticmethod
    def _read_file(p: Path) -> str:
        """Extract text from a file (supports .md, .txt, .pdf)."""
        if p.suffix.lower() in (".md", ".txt"):
            return p.read_text(encoding="utf-8")
        if p.suffix.lower() == ".pdf":
            with pymupdf.open(str(p)) as doc:
                return "\n".join(page.get_text() for page in doc)
        return ""

    def rebuild(self) -> int:
        paths = sorted(
            f for f in self._docs.rglob("*") if f.suffix.lower() in self._ALL_EXTS
        )
        meta: list[ChunkMeta] = []
        index = faiss.IndexFlatIP(self._dim)

        if paths:
            texts: list[str] = []
            for p in paths:
                doc = self._read_file(p)
                if not doc.strip():
                    continue
                chunks = self._chunk(doc) or [doc]
                for ci, ct in enumerate(chunks):
                    texts.append(
                        f"{p.stem} {p.stem} {p.relative_to(self._docs)} {ct} {ct[:400]}"
                    )
                    meta.append(
                        {
                            "title": p.stem,
                            "relative_path": str(p.relative_to(self._docs)),
                            "chunk_index": ci,
                            "chunk_text": ct,
                        }
                    )
            if texts:
                index.add(self._embedder.embed_texts(texts).astype("float32"))

        faiss.write_index(index, str(self._index_path))
        self._meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return len(paths)

    # -- search --------------------------------------------------------------

    def search(
        self,
        *,
        query: str,
        limit: int = 4,
        prefixes: list[str] | None = None,
        exclude_prefixes: list[str] | None = None,
    ) -> list[DocumentMatch]:
        if not self._index_path.exists() or not self._meta_path.exists():
            self.rebuild()

        meta = self._load_meta()
        if not meta:
            return []

        index = faiss.read_index(str(self._index_path))
        if index.d != self._dim:
            self.rebuild()
            meta = self._load_meta()
            if not meta:
                return []
            index = faiss.read_index(str(self._index_path))

        ranked: dict[tuple[str, int], float] = {}
        for variant in self._query_variants(query):
            qv = self._embedder.embed_query(variant).reshape(1, self._dim)
            top_k = len(meta) if prefixes else min(self._candidate_pool, len(meta))
            scores, indices = index.search(qv, top_k)
            for sem_score, idx in zip(scores[0], indices[0], strict=True):
                if idx < 0:
                    continue
                item = meta[idx]
                if prefixes and not any(
                    item["relative_path"].startswith(p) for p in prefixes
                ):
                    continue
                if exclude_prefixes and any(
                    item["relative_path"].startswith(p)
                    for p in exclude_prefixes
                ):
                    continue
                lex = self._lexical_score(variant, item)
                combined = float(sem_score) + lex
                key = (item["relative_path"], item["chunk_index"])
                if key not in ranked or combined > ranked[key]:
                    ranked[key] = combined

        results: list[DocumentMatch] = []
        for rp, ci in sorted(ranked, key=ranked.__getitem__, reverse=True):
            sc = ranked[(rp, ci)]
            if sc <= 0.0:
                continue
            item = next(
                m for m in meta if m["relative_path"] == rp and m["chunk_index"] == ci
            )
            results.append(
                DocumentMatch(
                    title=item["title"],
                    relative_path=rp,
                    score=sc,
                    chunk_index=ci,
                    excerpt=self._excerpt(meta, rp, ci),
                    mailbox=self._mailbox(rp),
                )
            )
            if len(results) >= limit:
                break
        return results

    def read_chunk(self, *, relative_path: str, chunk_index: int | None) -> str:
        if chunk_index is None:
            p = self._docs / relative_path
            return p.read_text(encoding="utf-8") if p.exists() else ""
        for item in self._load_meta():
            if (
                item["relative_path"] == relative_path
                and item["chunk_index"] == chunk_index
            ):
                return item["chunk_text"]
        return ""

    # -- internal ------------------------------------------------------------

    def _load_meta(self) -> list[ChunkMeta]:
        return cast(
            list[ChunkMeta], json.loads(self._meta_path.read_text(encoding="utf-8"))
        )

    def _chunk(self, text: str) -> list[str]:
        tokens = text.split()
        if not tokens:
            return []
        step = max(1, self._chunk_size - self._chunk_overlap)
        chunks: list[str] = []
        for i in range(0, len(tokens), step):
            c = " ".join(tokens[i : i + self._chunk_size]).strip()
            if c:
                chunks.append(c)
            if i + self._chunk_size >= len(tokens):
                break
        return chunks

    def _excerpt(self, meta: list[ChunkMeta], rp: str, ci: int) -> str:
        parts = [
            m["chunk_text"]
            for m in meta
            if m["relative_path"] == rp and ci - 1 <= m["chunk_index"] <= ci + 1
        ]
        return " ".join(parts)[:500].strip()

    def _query_variants(self, query: str) -> list[str]:
        tokens = _normalize_tokens(query)
        variants = [query]
        tok_q = " ".join(tokens)
        if tok_q and tok_q != query:
            variants.append(tok_q)
        expanded = list(tokens)
        for t in tokens:
            expanded.extend(QUERY_SYNONYMS.get(t, ()))
        exp_q = " ".join(dict.fromkeys(expanded))
        if exp_q and exp_q not in variants:
            variants.append(exp_q)
        return variants

    def _lexical_score(self, query: str, item: ChunkMeta) -> float:
        qt = set(_normalize_tokens(query))
        if not qt:
            return 0.0
        tt = set(
            _normalize_tokens(
                f"{item['title']} {item['relative_path']} {item['chunk_text'][:600]}"
            )
        )
        if not tt:
            return 0.0
        return len(qt & tt) / len(qt) * 0.35

    @staticmethod
    def _mailbox(rp: str) -> str | None:
        if "/sent/" in f"/{rp}":
            return "sent"
        if "/received/" in f"/{rp}":
            return "received"
        return None


def _normalize_tokens(text: str) -> list[str]:
    chars = [c.lower() if c.isalnum() else " " for c in text]
    return [t for t in "".join(chars).split() if len(t) > 2]
