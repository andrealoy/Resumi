"""Helpers to load synced Gmail markdown files for the UI."""

from pathlib import Path


class MailLoader:
    def __init__(self, *, docs_root: str) -> None:
        self._docs = Path(docs_root)

    def list_email_files(self) -> list[str]:
        paths = sorted((self._docs / "from_gmail").rglob("*.md"), reverse=True)
        return [str(p.relative_to(self._docs)) for p in paths]

    def read_email_file(self, relative_path: str) -> str:
        path = self._docs / relative_path
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")