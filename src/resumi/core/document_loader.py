"""Handle user document uploads and ingestion."""

from pathlib import Path
from typing import Any

from resumi.core.document_store import DocumentStore
from resumi.core.embedding import FaissKnowledgeBase


class DocumentLoader:
    def __init__(
        self,
        *,
        docs_root: str,
        knowledge_base: FaissKnowledgeBase,
        store: DocumentStore | None = None,
    ) -> None:
        self._docs = Path(docs_root)
        self._kb = knowledge_base
        self._store = store

    def save_files(self, files: list[Any], folder_name: str) -> str:
        if not folder_name.strip():
            return "Veuillez fournir un nom de dossier."

        dest = self._docs / folder_name.strip()
        dest.mkdir(parents=True, exist_ok=True)

        saved = 0
        for f in files or []:
            if not hasattr(f, "name"):
                continue
            src_path = Path(f.name)
            if not src_path.exists():
                continue
            target = dest / src_path.name
            content = src_path.read_bytes()
            text = content.decode("utf-8", errors="ignore")
            content_hash = DocumentStore.hash_content(text)
            if self._store and self._store.exists(content_hash=content_hash):
                continue
            target.write_bytes(content)
            rel = str(target.relative_to(self._docs))
            if self._store:
                self._store.add(
                    doc_type="doc",
                    source="upload",
                    title=src_path.name,
                    file_path=rel,
                    content_hash=content_hash,
                )
            saved += 1

        self._kb.rebuild()
        return f"{saved} fichier(s) ajoutés et indexés dans '{folder_name.strip()}'."
