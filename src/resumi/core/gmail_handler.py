"""Gmail OAuth reader + ingestion (fetch → save markdown → reindex)."""

import asyncio
import base64
import re
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow  # type: ignore[import-untyped]
from googleapiclient.discovery import Resource, build  # type: ignore[import-untyped]

from resumi.core.document_store import DocumentStore
from resumi.core.embedding import FaissKnowledgeBase

GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


class Direction(StrEnum):
    RECEIVED = "received"
    SENT = "sent"


class GmailHandler:
    def __init__(
        self,
        *,
        client_secrets_file: str,
        token_file: str,
        docs_root: str,
        knowledge_base: FaissKnowledgeBase,
        store: DocumentStore | None = None,
        query: str = "in:anywhere",
        user_id: str = "me",
    ) -> None:
        self._secrets = Path(client_secrets_file)
        self._token = Path(token_file)
        self._docs = Path(docs_root)
        self._kb = knowledge_base
        self._store = store
        self._query = query
        self._user_id = user_id
        self._oauth_flow: InstalledAppFlow | None = None
        self._oauth_state: str | None = None

    # -- public API ----------------------------------------------------------

    def is_connected(self) -> bool:
        """Return *True* if a valid (or refreshable) Gmail token exists."""
        if not self._token.exists():
            return False
        try:
            creds = Credentials.from_authorized_user_file(
                str(self._token), GMAIL_SCOPES
            )  # type: ignore[no-untyped-call]
            if creds and creds.valid:
                return True
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
                self._save_creds(creds)
                return True
        except Exception:
            return False
        return False

    def connect(self) -> bool:
        """Run the OAuth flow and return *True* on success."""
        try:
            self._credentials()
            return True
        except Exception:
            return False

    def has_client_secrets(self) -> bool:
        """Return *True* if the Gmail OAuth client JSON is available."""
        return self._secrets.exists()

    def begin_oauth(self, base_url: str) -> str:
        """Create a browser-friendly OAuth URL for local/Docker usage."""
        if not self._secrets.exists():
            raise FileNotFoundError(
                f"Missing Gmail OAuth client file at {self._secrets}"
            )
        flow = InstalledAppFlow.from_client_secrets_file(
            str(self._secrets), GMAIL_SCOPES
        )
        flow.redirect_uri = (
            f"{base_url.rstrip('/')}" "/api/v1/gmail/oauth/callback"
        )
        auth_url, state = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="consent",
        )
        self._oauth_flow = flow
        self._oauth_state = state
        return auth_url

    def finish_oauth(self, *, state: str, code: str) -> bool:
        """Complete the browser OAuth callback and persist the token."""
        if not self._oauth_flow or not self._oauth_state:
            return False
        if state != self._oauth_state:
            self._oauth_flow = None
            self._oauth_state = None
            return False
        try:
            self._oauth_flow.fetch_token(code=code)
            creds = self._oauth_flow.credentials
            self._save_creds(creds)
            return True
        except Exception:
            return False
        finally:
            self._oauth_flow = None
            self._oauth_state = None

    def last_sync_date(self) -> datetime | None:
        """Return the most recent sync timestamp found in saved mails, or *None*."""
        gmail_dir = self._docs / "from_gmail"
        if not gmail_dir.exists():
            return None
        latest: datetime | None = None
        for md in gmail_dir.rglob("*.md"):
            # filenames start with YYYYMMDDTHHMMSSz_
            m = re.match(r"(\d{8}T\d{6}Z)", md.name)
            if m:
                try:
                    dt = datetime.strptime(m.group(1), "%Y%m%dT%H%M%SZ").replace(
                        tzinfo=UTC
                    )
                    if latest is None or dt > latest:
                        latest = dt
                except ValueError:
                    continue
        return latest

    def sync_age_hours(self) -> float | None:
        """Hours since the last sync, or *None* if never synced."""
        last = self.last_sync_date()
        if last is None:
            return None
        return (datetime.now(UTC) - last).total_seconds() / 3600

    async def fetch_received(self, *, max_results: int = 100) -> list[dict[str, str]]:
        """Fetch received messages only."""
        return await self._fetch(max_results=max_results, direction=Direction.RECEIVED)

    async def fetch_sent(self, *, max_results: int = 100) -> list[dict[str, str]]:
        """Fetch sent messages only."""
        return await self._fetch(max_results=max_results, direction=Direction.SENT)

    def save_messages(self, messages: list[dict[str, str]], subdir: str) -> list[str]:
        """Save messages to disk (dedup via store) and return relative paths."""
        saved: list[str] = []
        for msg in messages:
            body = " ".join(msg["body"].split()).strip()
            content_hash = DocumentStore.hash_content(body)
            if self._store and self._store.exists(content_hash=content_hash):
                continue
            rel = self._save(msg, subdir)
            saved.append(rel)
            if self._store:
                self._store.add(
                    doc_type="mail",
                    source="gmail",
                    direction=subdir,
                    title=msg["subject"],
                    sender=msg.get("from"),
                    recipient=msg.get("to"),
                    file_path=rel,
                    content_hash=content_hash,
                )
        return saved

    def reindex(self) -> int:
        """Rebuild the FAISS index and return the number of indexed documents."""
        return self._kb.rebuild()

    async def sync(
        self,
        *,
        received_max: int = 100,
        sent_max: int = 100,
    ) -> dict[str, object]:
        received = await self.fetch_received(max_results=received_max)
        sent = await self.fetch_sent(max_results=sent_max)

        saved: list[str] = []
        saved.extend(self.save_messages(received, "received"))
        saved.extend(self.save_messages(sent, "sent"))

        indexed = self.reindex()
        return {
            "synced": len(saved),
            "received": len(received),
            "sent": len(sent),
            "indexed": indexed,
            "documents": saved,
        }

    # -- Gmail API -----------------------------------------------------------

    async def _fetch(
        self,
        *,
        max_results: int,
        direction: Direction | None = None,
    ) -> list[dict[str, str]]:
        return await asyncio.to_thread(self._fetch_sync, max_results, direction)

    def _fetch_sync(
        self, max_results: int, direction: Direction | None
    ) -> list[dict[str, str]]:
        service = build(
            "gmail", "v1", credentials=self._credentials(), cache_discovery=False
        )
        ids = self._list_ids(service, max_results, direction)
        return [self._get_message(service, mid) for mid in ids]

    def _credentials(self) -> Credentials:
        creds: Credentials | None = None
        if self._token.exists():
            creds = Credentials.from_authorized_user_file(
                str(self._token), GMAIL_SCOPES
            )  # type: ignore[no-untyped-call]
        if creds and creds.valid:
            return creds
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            self._save_creds(creds)
            return creds
        if not self._secrets.exists():
            raise FileNotFoundError(
                f"Missing Gmail OAuth client file at {self._secrets}"
            )
        flow = InstalledAppFlow.from_client_secrets_file(
            str(self._secrets), GMAIL_SCOPES
        )
        creds = flow.run_local_server(port=0)
        self._save_creds(creds)
        return creds

    def _save_creds(self, creds: Credentials) -> None:
        self._token.parent.mkdir(parents=True, exist_ok=True)
        self._token.write_text(creds.to_json(), encoding="utf-8")  # type: ignore[no-untyped-call]

    def _list_ids(
        self, service: Resource, max_results: int, direction: Direction | None
    ) -> list[str]:
        ids: list[str] = []
        page_token: str | None = None
        q = self._build_query(direction)
        while len(ids) < max_results:
            resp = (
                service.users()
                .messages()
                .list(
                    userId=self._user_id,
                    q=q,
                    maxResults=min(500, max_results - len(ids)),
                    pageToken=page_token,
                )
                .execute()
            )
            for m in resp.get("messages", []):
                if isinstance(m.get("id"), str):
                    ids.append(m["id"])
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
        return ids[:max_results]

    def _build_query(self, direction: Direction | None) -> str:
        parts = [self._query.strip()] if self._query.strip() else []
        if direction == Direction.SENT:
            parts.append("in:sent")
        elif direction == Direction.RECEIVED:
            parts.append("-in:sent")
        return " ".join(parts)

    def _get_message(self, service: Resource, mid: str) -> dict[str, str]:
        msg = (
            service.users()
            .messages()
            .get(userId=self._user_id, id=mid, format="full")
            .execute()
        )
        payload = msg.get("payload", {})
        headers = payload.get("headers", [])
        labels = msg.get("labelIds", [])
        direction = (
            "sent" if isinstance(labels, list) and "SENT" in labels else "received"
        )
        return {
            "id": mid,
            "subject": self._header(headers, "Subject") or "No subject",
            "from": self._header(headers, "From") or "",
            "to": self._header(headers, "To") or "",
            "direction": direction,
            "body": self._body(payload),
        }

    @staticmethod
    def _header(headers: list[object], name: str) -> str | None:
        for h in headers:
            if (
                isinstance(h, dict)
                and h.get("name") == name
                and isinstance(h.get("value"), str)
            ):
                value: str = h["value"]
                return value
        return None

    def _body(self, payload: dict[str, object]) -> str:
        mime = payload.get("mimeType")
        body = payload.get("body")
        if mime == "text/plain" and isinstance(body, dict):
            return self._decode(body.get("data"))
        parts = payload.get("parts")
        if isinstance(parts, list):
            for part in parts:
                if isinstance(part, dict):
                    text = self._body(part)
                    if text:
                        return text
        if isinstance(body, dict):
            return self._decode(body.get("data"))
        return ""

    @staticmethod
    def _decode(raw: object) -> str:
        if not isinstance(raw, str) or not raw:
            return ""
        return (
            base64.urlsafe_b64decode(raw.encode("utf-8"))
            .decode("utf-8", errors="ignore")
            .strip()
        )

    # -- document storage ----------------------------------------------------

    def _save(self, msg: dict[str, str], subdir: str) -> str:
        dest = self._docs / "from_gmail" / subdir
        dest.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        slug = _slugify(msg["subject"])
        path = dest / f"{ts}_{slug}.md"
        sender = msg["from"] or "me"
        recipient = msg["to"] or "unknown"
        subj = msg["subject"]
        header = f"From: {sender}\nTo: {recipient}\nSubject: {subj}\n\n"
        body = " ".join(msg["body"].split()).strip()
        path.write_text(f"# {msg['subject']}\n\n{header}{body}\n", encoding="utf-8")
        return str(path.relative_to(self._docs))


def _slugify(value: str, max_length: int = 80) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    slug = normalized.strip("-") or "document"
    if len(slug) > max_length:
        slug = slug[:max_length].rstrip("-")
    return slug
