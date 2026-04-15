from fastapi.testclient import TestClient

import resumi.main as main
from resumi.main import app


def test_health():
    client = TestClient(app)
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_gmail_connect_route_redirects(monkeypatch):
    class DummyGmail:
        def begin_oauth(self, base_url: str) -> str:
            assert base_url.startswith("http://testserver")
            return "https://accounts.google.com/test-auth"

    client = TestClient(app)
    monkeypatch.setattr(main, "_gmail", lambda: DummyGmail())

    resp = client.get("/api/v1/gmail/connect", follow_redirects=False)
    assert resp.status_code == 307
    assert resp.headers["location"] == "https://accounts.google.com/test-auth"


def test_gmail_oauth_callback_saves_token(monkeypatch):
    class DummyGmail:
        def finish_oauth(self, *, state: str, code: str) -> bool:
            return state == "ok-state" and code == "ok-code"

    client = TestClient(app)
    monkeypatch.setattr(main, "_gmail", lambda: DummyGmail())

    resp = client.get("/api/v1/gmail/oauth/callback?state=ok-state&code=ok-code")
    assert resp.status_code == 200
    assert "Gmail connecté" in resp.text
