"""SQLite-backed metadata store for all ingested documents."""

import hashlib
import sqlite3
from datetime import UTC, datetime
from pathlib import Path


class DocumentStore:
    """Lightweight metadata catalogue stored in a single SQLite file."""

    def __init__(self, db_path: str = ".data/documents.db") -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    # -- schema --------------------------------------------------------------

    def _create_tables(self) -> None:
        self._conn.executescript(
            """\
            CREATE TABLE IF NOT EXISTS documents (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                type        TEXT NOT NULL,       -- mail, audio, doc
                source      TEXT NOT NULL,       -- gmail, upload, micro …
                direction   TEXT,                -- received / sent (mails)
                title       TEXT NOT NULL,
                sender      TEXT,
                recipient   TEXT,
                date        TEXT NOT NULL,        -- ISO-8601
                file_path   TEXT NOT NULL UNIQUE, -- relative to docs_root
                content_hash TEXT NOT NULL,
                tags        TEXT DEFAULT '',
                category    TEXT DEFAULT '',
                created_at  TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_doc_type ON documents(type);
            CREATE INDEX IF NOT EXISTS idx_doc_hash ON documents(content_hash);
            CREATE INDEX IF NOT EXISTS idx_doc_date ON documents(date);
            """
        )

    # -- public API ----------------------------------------------------------

    def exists(self, *, content_hash: str) -> bool:
        """Return True if a document with this content hash is already stored."""
        row = self._conn.execute(
            "SELECT 1 FROM documents WHERE content_hash = ?", (content_hash,)
        ).fetchone()
        return row is not None

    def exists_by_path(self, file_path: str) -> bool:
        """Return True if a document with this file_path is already stored."""
        row = self._conn.execute(
            "SELECT 1 FROM documents WHERE file_path = ?", (file_path,)
        ).fetchone()
        return row is not None

    def add(
        self,
        *,
        doc_type: str,
        source: str,
        title: str,
        file_path: str,
        content_hash: str,
        direction: str | None = None,
        sender: str | None = None,
        recipient: str | None = None,
        date: str | None = None,
        tags: str = "",
        category: str = "",
    ) -> int:
        """Insert a new document row and return its id."""
        now = datetime.now(UTC).isoformat()
        cur = self._conn.execute(
            """\
            INSERT OR IGNORE INTO documents
                (type, source, direction, title, sender, recipient,
                 date, file_path, content_hash, tags, category, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc_type,
                source,
                direction,
                title,
                sender,
                recipient,
                date or now,
                file_path,
                content_hash,
                tags,
                category,
                now,
            ),
        )
        self._conn.commit()
        return cur.lastrowid or 0

    def count(self, *, doc_type: str | None = None) -> int:
        if doc_type:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM documents WHERE type = ?", (doc_type,)
            ).fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()
        return row[0] if row else 0

    def last_date(self, *, doc_type: str | None = None) -> str | None:
        """Return the most recent document date, or None."""
        if doc_type:
            row = self._conn.execute(
                "SELECT MAX(date) FROM documents WHERE type = ?", (doc_type,)
            ).fetchone()
        else:
            row = self._conn.execute("SELECT MAX(date) FROM documents").fetchone()
        return row[0] if row and row[0] else None

    def search(
        self,
        *,
        doc_type: str | None = None,
        source: str | None = None,
        direction: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, object]]:
        """Return documents matching filters, newest first."""
        clauses: list[str] = []
        params: list[str] = []
        if doc_type:
            clauses.append("type = ?")
            params.append(doc_type)
        if source:
            clauses.append("source = ?")
            params.append(source)
        if direction:
            clauses.append("direction = ?")
            params.append(direction)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM documents {where} ORDER BY date DESC LIMIT ?",  # noqa: S608
            (*params, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def update_category(self, doc_id: int, category: str) -> None:
        """Update the category of a document by id."""
        self._conn.execute(
            "UPDATE documents SET category = ? WHERE id = ?", (category, doc_id)
        )
        self._conn.commit()

    def uncategorized_mails(self, limit: int = 200) -> list[dict[str, object]]:
        """Return mails that have no category yet."""
        rows = self._conn.execute(
            "SELECT * FROM documents WHERE type = 'mail'"
            " AND (category = '' OR category IS NULL) "
            "ORDER BY date DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def uncategorized_docs(self, limit: int = 200) -> list[dict[str, object]]:
        """Return uploaded docs that have no category yet."""
        rows = self._conn.execute(
            "SELECT * FROM documents WHERE type = 'doc'"
            " AND (category = '' OR category IS NULL) "
            "ORDER BY date DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self._conn.close()

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def hash_content(text: str) -> str:
        """SHA-256 hex digest of the given text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
