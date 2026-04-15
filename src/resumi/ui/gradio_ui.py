"""Gradio Blocks interface."""

import os
import tempfile
from collections.abc import Generator
from datetime import date, timedelta
from typing import cast

import calendar as pycalendar

import gradio as gr
import pandas as pd

from resumi.core.agent import Agent
from resumi.core.audio_handler import start_recording, stop_recording, transcribe_file
from resumi.core.calendar import CALENDAR_FILE
from resumi.core.document_loader import DocumentLoader
from resumi.core.document_store import DocumentStore
from resumi.core.gmail_handler import GmailHandler
from resumi.core.mail_loader import MailLoader
from resumi.ui.chat import ask, run_async

# Threshold (hours) beyond which we consider the mail sync stale.
_STALE_HOURS = 24.0


def _sync_age_label(gmail_handler: GmailHandler) -> str:
    """Human-readable description of mail freshness."""
    age = gmail_handler.sync_age_hours()
    if age is None:
        return "Aucune synchronisation précédente."
    if age < 1:
        return "Dernière synchronisation : il y a moins d'une heure."
    if age < 24:
        return f"Dernière synchronisation : il y a {int(age)} h."
    days = int(age // 24)
    return f"Dernière synchronisation : il y a {days} jour{'s' if days > 1 else ''}."


def _parse_sync_count(text: str) -> int | None:
    """Parse a mail count from a follow-up message (e.g. '50', 'ok')."""
    t = text.strip().casefold()
    if t in ("ok", "oui", "défaut", "defaut", "default", "d", ""):
        return 100
    if t in ("tous", "tout", "all", "max", "maximum"):
        return 500
    try:
        return max(1, min(int(t), 1000))
    except ValueError:
        return None


def _extract_sync_count(text: str) -> int | None:
    """Extract a number from a sync request like 'synchronise 200 mails'."""
    for word in text.split():
        try:
            n = int(word)
            if 1 <= n <= 1000:
                return n
        except ValueError:
            continue
    return None


# Short confirmations that mean "go ahead" after a sync/connect prompt.
_CONFIRM_WORDS = frozenset({
    "vas-y", "vas y", "vashy", "go", "oui", "ok", "d'accord",
    "d accord", "allez", "lance", "c'est parti", "c est parti",
    "let's go", "lets go", "yep", "yes", "ouais", "envoie",
    "parfait", "top", "génial", "genial", "allons-y", "allons y",
})

# Phrases in the last assistant message that indicate a sync suggestion.
_SYNC_HINT_PHRASES = (
    "synchroniser",
    "synchronise",
    "sync",
    "récupérer tes mails",
    "recuperer tes mails",
    "importe",
)


def _is_sync_confirmation(
    text: str, history: list[dict[str, str]]
) -> bool:
    """Return True if *text* is a short go-ahead after a sync-related message."""
    low = text.strip().casefold()
    # Must be a short confirmation phrase
    if low not in _CONFIRM_WORDS and not any(
        low.startswith(w) for w in _CONFIRM_WORDS
    ):
        return False
    # Check if the last assistant message hints at sync
    for h in reversed(history[-4:]):
        if h.get("role") == "assistant":
            content = h["content"].casefold()
            return any(p in content for p in _SYNC_HINT_PHRASES)
    return False


def _run_sync_steps(
    gmail_handler: GmailHandler,
    agent: Agent,
    history: list[dict[str, str]],
    *,
    max_results: int = 100,
) -> Generator[list[dict[str, str]], None, None]:
    """Generator that yields chat history updates during a step-by-step Gmail sync."""
    half = max_results // 2
    rest = max_results - half

    def _msg(text: str) -> dict[str, str]:
        return {"role": "assistant", "content": text}

    # Step 1 — fetch received
    history = history + [_msg(f"📥 [1/5] Récupération des mails reçus (max {half})…")]
    yield history
    try:
        received = run_async(gmail_handler.fetch_received(max_results=half))
    except Exception as exc:
        yield history + [_msg(f"❌ Erreur récupération mails reçus : {exc}")]
        return
    history = history + [_msg(f"✅ **{len(received)}** mails reçus récupérés.")]
    yield history

    # Step 2 — fetch sent
    history = history + [_msg(f"📤 [2/5] Récupération des mails envoyés (max {rest})…")]
    yield history
    try:
        sent = run_async(gmail_handler.fetch_sent(max_results=rest))
    except Exception as exc:
        yield history + [_msg(f"❌ Erreur récupération mails envoyés : {exc}")]
        return
    history = history + [_msg(f"✅ **{len(sent)}** mails envoyés récupérés.")]
    yield history

    # Step 3 — save to disk
    history = history + [_msg("💾 [3/5] Sauvegarde des mails sur le disque…")]
    yield history
    saved_received = gmail_handler.save_messages(received, "received")
    saved_sent = gmail_handler.save_messages(sent, "sent")
    total_saved = len(saved_received) + len(saved_sent)
    history = history + [_msg(f"✅ **{total_saved}** nouveaux mails sauvegardés.")]
    yield history

    # Step 4 — reindex
    history = history + [_msg("🔍 [4/5] Indexation des documents…")]
    yield history
    indexed = gmail_handler.reindex()
    history = history + [_msg(f"✅ **{indexed}** documents indexés.")]
    yield history

    # Step 5 — classify uncategorized mails
    history = history + [_msg("🏷️ [5/5] Classification automatique des mails…")]
    yield history
    try:
        total_uncategorized = (
            len(agent._store.uncategorized_mails()) if agent._store else 0
        )
        classified = run_async(agent.classify_uncategorized_mails())
        failed = total_uncategorized - classified
        if failed > 0:
            history = history + [
                _msg(
                    f"⚠️ **{classified}** mails classifiés, "
                    f"**{failed}** en échec "
                    "(erreurs API / rate-limit). "
                    "Relance la synchronisation "
                    "pour réessayer.\n\n"
                    "Tu peux consulter l'onglet **Mails** pour voir les catégories, "
                    "ou me demander un brouillon de réponse dans le chat."
                )
            ]
        else:
            history = history + [
                _msg(
                    f"✅ **{classified}** mails classifiés.\n\n"
                    "Tu peux consulter l'onglet **Mails** pour voir les catégories, "
                    "ou me demander un brouillon de réponse dans le chat."
                )
            ]
    except Exception as exc:
        history = history + [_msg(f"⚠️ Classification partielle : {exc}")]
    yield history


def _build_mail_table(store: DocumentStore, direction: str) -> list[list[str]]:
    """Build table rows [[title, sender/recipient, category, date]] from the store."""
    mails = store.search(doc_type="mail", direction=direction, limit=100)
    rows: list[list[str]] = []
    for m in mails:
        title = str(m.get("title", ""))
        key = "sender" if direction == "received" else "recipient"
        person = str(m.get(key, ""))
        category = str(m.get("category", "")) or "—"
        date = str(m.get("date", ""))[:10]
        rows.append([title, person, category, date])
    return rows


def _build_doc_table(store: DocumentStore) -> list[list[str]]:
    """Build table rows [[title, category, date]] for uploaded docs."""
    docs = store.search(doc_type="doc", limit=100)
    rows: list[list[str]] = []
    for d in docs:
        if d.get("type") in ["doc", "audio"]:
            title = str(d.get("title", ""))
            category = str(d.get("category", "")) or "—"
            date = str(d.get("date", ""))[:10]
            rows.append([title, category, date])
    return rows


def _build_calendar_table() -> list[list[str]]:
    """Read the local calendar CSV for Gradio display."""
    if os.path.exists(CALENDAR_FILE):
        try:
            df = pd.read_csv(CALENDAR_FILE)
            return df.values.tolist()
        except Exception:
            return []
    return []


def _build_calendar_html(year: int | None = None, month: int | None = None) -> str:
    """Render an HTML monthly calendar with events from the CSV."""
    today = date.today()
    if year is None:
        year = today.year
    if month is None:
        month = today.month

    # Load events
    events: dict[str, list[tuple[str, str]]] = {}  # "YYYY-MM-DD" -> [(heure, label)]
    if os.path.exists(CALENDAR_FILE):
        try:
            df = pd.read_csv(CALENDAR_FILE)
            for _, row in df.iterrows():
                d = str(row.get("Date", ""))
                h = str(row.get("Heure", ""))
                ev = str(row.get("Événement", row.get("Evenement", "")))
                events.setdefault(d, []).append((h, ev))
        except Exception:
            pass

    # Month navigation
    prev_month = month - 1 if month > 1 else 12
    prev_year = year if month > 1 else year - 1
    next_month = month + 1 if month < 12 else 1
    next_year = year if month < 12 else year + 1

    month_names_fr = [
        "", "janvier", "février", "mars", "avril", "mai", "juin",
        "juillet", "août", "septembre", "octobre", "novembre", "décembre",
    ]
    day_names = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]

    # Calendar grid
    cal = pycalendar.Calendar(firstweekday=0)
    weeks = cal.monthdayscalendar(year, month)

    html = f"""
    <div class="cal-container">
      <div class="cal-header">
        <span class="cal-title">{month_names_fr[month].capitalize()} {year}</span>
      </div>
      <table class="cal-grid">
        <thead><tr>
    """
    for d in day_names:
        html += f'<th class="cal-dayname">{d}</th>'
    html += "</tr></thead><tbody>"

    for week in weeks:
        html += "<tr>"
        for day in week:
            if day == 0:
                html += '<td class="cal-cell cal-empty"></td>'
                continue

            day_str = f"{year}-{month:02d}-{day:02d}"
            is_today = (year == today.year and month == today.month and day == today.day)
            day_events = events.get(day_str, [])

            cls = "cal-cell"
            if is_today:
                cls += " cal-today"
            if day_events:
                cls += " cal-has-event"

            html += f'<td class="{cls}"><div class="cal-day-num">{day}</div>'
            for h, ev in day_events[:3]:  # max 3 per cell
                html += (
                    f'<div class="cal-event" title="{h} — {ev}">'
                    f'<span class="cal-event-time">{h}</span> {ev}</div>'
                )
            if len(day_events) > 3:
                html += f'<div class="cal-event cal-more">+{len(day_events) - 3} autres</div>'
            html += "</td>"
        html += "</tr>"

    html += "</tbody></table></div>"
    return html


# --- Thème de l'interface
theme = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#f3e8ff",
        c100="#e9d5ff",
        c200="#d8b4fe",
        c300="#c084fc",
        c400="#a855f7",
        c500="#9333ea",
        c600="#7c3aed",
        c700="#6d28d9",
        c800="#5b21b6",
        c900="#4c1d95",
        c950="#3b0764",
    ),
    secondary_hue="zinc",
    neutral_hue="zinc",
    spacing_size="md",
    radius_size="lg",
    font=(
        gr.themes.Font("Helvetica Neue"),
        gr.themes.Font("Helvetica"),
        gr.themes.Font("Arial"),
        gr.themes.Font("sans-serif"),
    ),
    font_mono=(
        gr.themes.GoogleFont("JetBrains Mono"),
        gr.themes.Font("Fira Code"),
        gr.themes.Font("monospace"),
    ),
).set(
    # --- Background ---
    body_background_fill="#0f0f14",
    body_background_fill_dark="#0f0f14",
    background_fill_primary="#16161d",
    background_fill_primary_dark="#16161d",
    background_fill_secondary="#1c1c27",
    background_fill_secondary_dark="#1c1c27",
    block_background_fill="#1c1c27",
    block_background_fill_dark="#1c1c27",
    block_border_color="#2a2a3a",
    block_border_color_dark="#2a2a3a",
    block_label_background_fill="#1c1c27",
    block_label_background_fill_dark="#1c1c27",
    block_title_text_color="#e4e4ed",
    block_title_text_color_dark="#e4e4ed",
    block_label_text_color="#a1a1b5",
    block_label_text_color_dark="#a1a1b5",
    # --- Text ---
    body_text_color="#e4e4ed",
    body_text_color_dark="#e4e4ed",
    body_text_color_subdued="#8888a0",
    body_text_color_subdued_dark="#8888a0",
    # --- Borders ---
    border_color_primary="#2a2a3a",
    border_color_primary_dark="#2a2a3a",
    input_background_fill="#1c1c27",
    input_background_fill_dark="#1c1c27",
    input_border_color="#2a2a3a",
    input_border_color_dark="#2a2a3a",
    input_border_color_focus="#7c3aed",
    input_border_color_focus_dark="#7c3aed",
    # --- Buttons ---
    button_primary_background_fill="#7c3aed",
    button_primary_background_fill_dark="#7c3aed",
    button_primary_background_fill_hover="#6d28d9",
    button_primary_background_fill_hover_dark="#6d28d9",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    button_secondary_background_fill="#2a2a3a",
    button_secondary_background_fill_dark="#2a2a3a",
    button_secondary_background_fill_hover="#353548",
    button_secondary_background_fill_hover_dark="#353548",
    button_secondary_text_color="#e4e4ed",
    button_secondary_text_color_dark="#e4e4ed",
    button_cancel_background_fill="#dc2626",
    button_cancel_background_fill_dark="#dc2626",
    button_cancel_background_fill_hover="#b91c1c",
    button_cancel_background_fill_hover_dark="#b91c1c",
    # --- Spacing ---
    layout_gap="14px",
    container_radius="10px",
    shadow_drop="0 2px 8px rgba(0,0,0,0.35)",
    shadow_drop_lg="0 4px 16px rgba(0,0,0,0.4)",
    # --- Tables ---
    table_border_color="#2a2a3a",
    table_border_color_dark="#2a2a3a",
    table_even_background_fill="#1c1c27",
    table_even_background_fill_dark="#1c1c27",
    table_odd_background_fill="#16161d",
    table_odd_background_fill_dark="#16161d",
    table_row_focus="#2a2a3a",
    table_row_focus_dark="#2a2a3a",
)

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,wght@0,300;0,400;0,500;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap');

/* Font stacks */
:root {
    --font-serif: 'Source Serif 4',
        'Palatino Linotype', 'Book Antiqua',
        Palatino, Georgia, serif;
}

/* Global */
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
}
footer { display: none !important; }

/* Headings */
h1, h2, h3 {
    color: #e4e4ed !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em;
}
h1 { font-size: 1.6rem !important; }

/* Buttons */
button.primary {
    transition: all 0.2s ease !important;
    box-shadow: 0 0 12px rgba(124, 58, 237, 0.25) !important;
}
button.primary:hover {
    box-shadow: 0 0 20px rgba(124, 58, 237, 0.4) !important;
    transform: translateY(-1px);
}
button.secondary {
    transition: all 0.2s ease !important;
}

/* Upload zone */
.dashed-upload {
    border: 2px dashed #3b3b50 !important;
    border-radius: 10px !important;
    background: rgba(124, 58, 237, 0.03) !important;
    transition: border-color 0.2s ease !important;
}
.dashed-upload:hover {
    border-color: #7c3aed !important;
}
.dashed-upload .file-preview {
    border: none !important;
}

/* Tabs */
.tabs > .tab-nav > button {
    color: #8888a0 !important;
    font-weight: 500 !important;
    border: none !important;
    transition: color 0.2s ease !important;
}
.tabs > .tab-nav > button.selected {
    color: #c084fc !important;
    border-bottom: 2px solid #7c3aed !important;
}

/* Inputs */
textarea, input[type="text"] {
    caret-color: #7c3aed !important;
}

/* Chatbot messages */
.message.bot, .message.user {
    font-family: var(--font-serif) !important;
    font-size: 1.01rem !important;
    line-height: 1.7 !important;
    letter-spacing: 0.01em !important;
}
.message.bot {
    background: #1c1c27 !important;
    border: 1px solid #2a2a3a !important;
}
.message.user {
    background: rgba(124, 58, 237, 0.12) !important;
    border: 1px solid rgba(124, 58, 237, 0.25) !important;
}

/* Accordion */
.accordion {
    border-color: #2a2a3a !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0f0f14; }
::-webkit-scrollbar-thumb {
    background: #2a2a3a;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: #3b3b50; }

/* Gmail status dot */
.gmail-status-row {
    display: flex !important;
    align-items: center !important;
    justify-content: flex-end !important;
    gap: 8px !important;
    padding: 4px 0 !important;
}
.gmail-dot {
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
}
.gmail-dot.connected { background: #22c55e; box-shadow: 0 0 6px #22c55e88; }
.gmail-dot.disconnected { background: #ef4444; box-shadow: 0 0 6px #ef444488; }

/* Logo */
.resumi-logo {
    font-family: var(--font-serif);
    font-size: 1.8rem;
    font-weight: 600;
    color: #e4e4ed;
    letter-spacing: -0.02em;
    line-height: 1.2;
}
.resumi-dot {
    color: #7c3aed;
    font-weight: 700;
}
.resumi-tagline {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-size: 0.88rem;
    color: #8888a0;
    margin-top: 2px;
    letter-spacing: 0.01em;
}

/* Send button */
.send-btn {
    min-width: 80px !important;
    height: 42px !important;
}

/* ── Calendar visual ── */
.cal-container {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
}
.cal-header {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 16px;
    padding: 12px 0 8px;
}
.cal-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: #e4e4ed;
    letter-spacing: -0.01em;
}
.cal-grid {
    width: 100%;
    border-collapse: separate;
    border-spacing: 4px;
    table-layout: fixed;
}
.cal-dayname {
    color: #8888a0;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    text-align: center;
    padding: 6px 0;
}
.cal-cell {
    background: #1c1c27;
    border: 1px solid #2a2a3a;
    border-radius: 8px;
    vertical-align: top;
    padding: 6px 8px;
    min-height: 80px;
    height: 80px;
    transition: border-color 0.15s ease;
}
.cal-cell:hover {
    border-color: #3b3b50;
}
.cal-empty {
    background: transparent;
    border-color: transparent;
}
.cal-today {
    border-color: #7c3aed !important;
    box-shadow: 0 0 8px rgba(124, 58, 237, 0.25);
}
.cal-has-event {
    background: #1e1c2a;
}
.cal-day-num {
    font-size: 0.82rem;
    font-weight: 600;
    color: #a1a1b5;
    margin-bottom: 4px;
}
.cal-today .cal-day-num {
    color: #c084fc;
    font-weight: 700;
}
.cal-event {
    font-size: 0.7rem;
    line-height: 1.3;
    color: #e4e4ed;
    background: rgba(124, 58, 237, 0.18);
    border-left: 2px solid #7c3aed;
    border-radius: 4px;
    padding: 2px 6px;
    margin-bottom: 2px;
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
}
.cal-event-time {
    color: #c084fc;
    font-weight: 600;
    margin-right: 3px;
}
.cal-more {
    color: #8888a0;
    background: transparent;
    border-left-color: #3b3b50;
    font-style: italic;
}
.cal-nav-row {
    display: flex;
    justify-content: center;
    gap: 8px;
    padding-top: 8px;
}
"""


# --- INTERFACE UI
def create_gradio_blocks(
    *,
    agent: Agent,
    mail_loader: MailLoader,
    document_loader: DocumentLoader,
    gmail_handler: GmailHandler,
    store: DocumentStore,
) -> gr.Blocks:

    # --- Container
    with gr.Blocks(title="Resumi", theme=theme, css=custom_css) as demo:
        # --- Header row: title + gmail button ---
        with gr.Row():
            with gr.Column(scale=6):
                gr.HTML(
                    '<div class="resumi-logo">'
                    "Resumi"
                    '<span class="resumi-dot">.</span>'
                    "</div>"
                    '<p class="resumi-tagline">'
                    "Synchroniser vos mails, "
                    "synthétiser un fichier audio "
                    "ou classer des documents."
                    "</p>"
                )
            with gr.Column(scale=1, min_width=180):
                connected = gmail_handler.is_connected()
                dot_cls = "connected" if connected else "disconnected"
                dot_label = "connecté" if connected else "déconnecté"
                gmail_status = gr.HTML(
                    f'<div class="gmail-status-row">'
                    f'<span class="gmail-dot {dot_cls}"></span>'
                    f'<span style="color:#a1a1b5;font-size:0.85rem">'
                    f"Gmail {dot_label}</span></div>",
                )
                gmail_connect_btn = gr.Button(
                    "Synchroniser" if connected else "Se connecter",
                    variant="primary" if not connected else "secondary",
                    size="sm",
                )
        with gr.Tabs():
            # ── Tab 1: Chat Assistant ──────────────────────────────────
            with gr.Tab("Chat"):
                # --- message de bienvenue
                initial_msgs = [
                    {
                        "role": "assistant",
                        "content": (
                            "Bonjour ! Que voulez-vous "
                            "faire aujourd'hui ?"
                        ),
                    }
                ]

                # --- chatbox
                chatbot = gr.Chatbot(
                    type="messages",
                    allow_tags=False,
                    value=initial_msgs,
                    label="",
                    show_label=False,
                )

                # --- message input + send + audio
                with gr.Row():
                    message = gr.Textbox(
                        placeholder="Demander à Resumi",
                        scale=8,
                        lines=1,
                        container=False,
                    )
                    btn_send = gr.Button(
                        "Envoyer",
                        variant="primary",
                        scale=1,
                        min_width=80,
                        elem_classes=["send-btn"],
                    )
                    btn_mic = gr.Button(
                        "🎙️",
                        variant="secondary",
                        scale=0,
                        min_width=50,
                    )
                    btn_stop = gr.Button(
                        "🛑",
                        variant="stop",
                        scale=0,
                        min_width=50,
                    )

                # --- others
                with gr.Accordion("Sources consultées", open=False):
                    sources = gr.JSON(label="Sources RAG", show_label=False)

                with gr.Accordion("🗓️ Calendrier", open=False):
                    calendar_table = gr.Dataframe(
                        headers=["Date", "Heure", "Événement"],
                        value=_build_calendar_table(),
                        interactive=False,
                        wrap=True,
                    )

                is_rec = gr.State(False)
                pending_sync = gr.State(False)

                # --- functions connexion to gmail
                _dot_html_ok = (
                    '<div class="gmail-status-row">'
                    '<span class="gmail-dot connected"></span>'
                    '<span style="color:#a1a1b5;'
                    'font-size:0.85rem">'
                    "Gmail connecté</span></div>"
                )
                _dot_html_ko = (
                    '<div class="gmail-status-row">'
                    '<span class="gmail-dot disconnected"></span>'
                    '<span style="color:#a1a1b5;'
                    'font-size:0.85rem">'
                    "Gmail déconnecté</span></div>"
                )
                _dot_html_sync = (
                    '<div class="gmail-status-row">'
                    '<span class="gmail-dot connected"></span>'
                    '<span style="color:#a1a1b5;'
                    'font-size:0.85rem">'
                    "Synchro…</span></div>"
                )

                def handle_gmail_btn(history: list[dict[str, str]]):
                    if not gmail_handler.is_connected():
                        if not gmail_handler.has_client_secrets():
                            yield (
                                history
                                + [{
                                    "role": "assistant",
                                    "content": (
                                        "⚠️ Le fichier OAuth Gmail est introuvable dans le conteneur.\n\n"
                                        "Relance Docker avec le dossier credentials monté sur /app/credentials, "
                                        "puis réessaie."
                                    ),
                                }],
                                _dot_html_ko,
                                gr.update(value="Se connecter"),
                            )
                            return

                        yield (
                            history
                            + [{
                                "role": "assistant",
                                "content": (
                                    "🔐 Pour connecter Gmail, ouvre ce lien dans ton navigateur : "
                                    "[Se connecter à Gmail](/api/v1/gmail/connect)\n\n"
                                    "Après validation Google, reviens ici puis reclique sur Synchroniser."
                                ),
                            }],
                            _dot_html_ko,
                            gr.update(value="Se connecter"),
                        )
                        return

                    # Lancement de la synchro
                    for new_history in _run_sync_steps(
                        gmail_handler, agent, history
                    ):
                        yield (
                            new_history,
                            _dot_html_sync,
                            gr.update(),
                        )
                    yield (
                        new_history,
                        _dot_html_ok,
                        gr.update(
                            value="Synchroniser",
                            variant="secondary",
                        ),
                    )

                gmail_connect_btn.click(
                    fn=handle_gmail_btn,
                    inputs=[chatbot],
                    outputs=[chatbot, gmail_status, gmail_connect_btn],
                )

                # Auto-sync on load if stale
                needs_auto_sync = connected and (
                    gmail_handler.sync_age_hours() is None
                    or (gmail_handler.sync_age_hours() or 0) >= _STALE_HOURS
                )

                def auto_sync_on_load(
                    history: list[dict[str, str]],
                ) -> Generator[
                    tuple[list[dict[str, str]], str], None, None
                ]:
                    if not needs_auto_sync:
                        yield history, _dot_html_ok
                        return
                    yield from handle_gmail_btn(history)

                # demo.load(
                #     fn=auto_sync_on_load,
                #     inputs=[chatbot],
                #     outputs=[chatbot, gmail_status],
                # )

                # --- functions responses from agents
                def respond(
                    text: str,
                    history: list[dict[str, str]],
                    is_pending_sync: bool,
                ) -> Generator[
                    tuple[
                        list[dict[str, str]],
                        str,
                        list[dict[str, object]],
                        bool,
                        list[list[str]],
                    ],
                    None,
                    None,
                ]:
                    # -- Follow-up to a sync count question ------
                    if is_pending_sync:
                        count = _parse_sync_count(text)
                        if count is not None:
                            history = history + [
                                {"role": "user", "content": text},
                            ]
                            for h in _run_sync_steps(
                                gmail_handler,
                                agent,
                                history,
                                max_results=count,
                            ):
                                yield h, "", [], False, _build_calendar_table()
                            return
                        # Not a valid count — reset and handle normally

                    # -- Contextual sync confirmation ("vas y", "oui"…)
                    if _is_sync_confirmation(text, history):
                        if not gmail_handler.is_connected():
                            updated = history + [
                                {"role": "user", "content": text},
                                {
                                    "role": "assistant",
                                    "content": (
                                        "⚠️ Gmail n'est pas connecté. "
                                        "Dis-moi « connecte Gmail » "
                                        "d'abord."
                                    ),
                                },
                            ]
                            yield updated, "", [], False, _build_calendar_table()
                            return
                        # Ask how many mails
                        updated = history + [
                            {"role": "user", "content": text},
                            {
                                "role": "assistant",
                                "content": (
                                    "📬 Combien de mails veux-tu "
                                    "télécharger ?\n\n"
                                    "Donne un nombre (ex: `50`, "
                                    "`200`) ou tape `ok` pour le "
                                    "défaut (100)."
                                ),
                            },
                        ]
                        yield updated, "", [], True, _build_calendar_table()
                        return

                    # -- Detect sync request from chat -----------
                    tool = agent._needs_tool(text)
                    if tool == "gmail_sync":
                        if not gmail_handler.is_connected():
                            updated = history + [
                                {"role": "user", "content": text},
                                {
                                    "role": "assistant",
                                    "content": (
                                        "⚠️ Gmail n'est pas connecté. "
                                        "Dis-moi « connecte Gmail » "
                                        "d'abord."
                                    ),
                                },
                            ]
                            yield updated, "", [], False, _build_calendar_table()
                            return

                        # If the user specified a count inline
                        count = _extract_sync_count(text)
                        if count is not None:
                            history = history + [
                                {"role": "user", "content": text},
                            ]
                            for h in _run_sync_steps(
                                gmail_handler,
                                agent,
                                history,
                                max_results=count,
                            ):
                                yield h, "", [], False, _build_calendar_table()
                            return

                        # Ask how many mails
                        updated = history + [
                            {"role": "user", "content": text},
                            {
                                "role": "assistant",
                                "content": (
                                    "📬 Combien de mails veux-tu "
                                    "télécharger ?\n\n"
                                    "Donne un nombre (ex: `50`, "
                                    "`200`) ou tape `ok` pour le "
                                    "défaut (100)."
                                ),
                            },
                        ]
                        yield updated, "", [], True, _build_calendar_table()
                        return

                    # -- Normal message --------------------------
                    result = ask(agent, text, history)
                    updated = history + [
                        {"role": "user", "content": text},
                        {
                            "role": "assistant",
                            "content": str(result["answer"]),
                        },
                    ]
                    yield (
                        updated,
                        "",
                        result.get("sources", []),  # type: ignore[arg-type]
                        False,
                        _build_calendar_table(),
                    )

                # --- functions mic
                def agent_voice_respond(transcript: str, history: list[dict[str, str]]):
                    if not transcript or transcript.startswith("("):
                        return history, [], _build_calendar_table()

                    result = ask(
                        agent, transcript, history[:-2]
                    )  # On enlève les 2 derniers (user + loading) pour l'appel

                    new_history = history[:-1] + [
                        {"role": "assistant", "content": str(result["answer"])}
                    ]
                    return new_history, result.get("sources", []), _build_calendar_table()

                # Un état caché pour stocker le texte transcrit entre les deux étapes
                temp_transcript = gr.State("")

                chat_event = message.submit(
                    respond,
                    inputs=[message, chatbot, pending_sync],
                    outputs=[chatbot, message, sources, pending_sync, calendar_table],
                )
                btn_send.click(
                    respond,
                    inputs=[message, chatbot, pending_sync],
                    outputs=[chatbot, message, sources, pending_sync, calendar_table],
                )

                def handle_mic(recording: bool, history: list[dict[str, str]]):
                    if not recording:
                        try:
                            start_recording()
                        except Exception as exc:
                            history = history + [
                                {
                                    "role": "assistant",
                                    "content": f"❌ Micro indisponible : {exc}",
                                },
                            ]
                            return (
                                False,
                                gr.update(value="🎙️", variant="secondary"),
                                history,
                                "",
                            )
                        return True, gr.update(value="⏹", variant="stop"), history, ""
                    else:
                        try:
                            _, transcript = stop_recording()
                        except Exception as exc:
                            history = history + [
                                {
                                    "role": "assistant",
                                    "content": f"❌ Erreur enregistrement : {exc}",
                                },
                            ]
                            return (
                                False,
                                gr.update(value="🎙️", variant="secondary"),
                                history,
                                "",
                            )
                        if not transcript or transcript.startswith("("):
                            return (
                                False,
                                gr.update(value="🎙️", variant="secondary"),
                                history,
                                "",
                            )

                        new_history = history + [
                            {"role": "user", "content": transcript},
                            {
                                "role": "assistant",
                                "content": "*En train de réfléchir...*",
                            },
                        ]
                        return (
                            False,
                            gr.update(value="🎙️", variant="secondary"),
                            new_history,
                            transcript,
                        )

                voice_event = btn_mic.click(
                    fn=handle_mic,
                    inputs=[is_rec, chatbot],
                    outputs=[is_rec, btn_mic, chatbot, temp_transcript],
                ).then(
                    fn=agent_voice_respond,
                    inputs=[temp_transcript, chatbot],
                    outputs=[chatbot, sources, calendar_table],
                )

                btn_stop.click(
                    fn=lambda: (
                        False,
                        gr.update(value="🎙️", variant="secondary", interactive=True),
                        gr.update(),
                    ),
                    outputs=[is_rec, btn_mic, btn_stop],
                    cancels=[chat_event, voice_event],
                )

            # ── Tab 2: Upload Documents ────────────────────────────────

            with gr.Tab("Documents"):
                # Zone d'ajout rétractable pour ne pas encombrer l'écran
                with gr.Accordion("Ajouter de nouveaux documents", open=False):
                    folder_input = gr.Textbox(
                        label="Dossier cible (obligatoire)",
                        placeholder="ex: factures_2024",
                        scale=2,
                    )

                    with gr.Row():
                        with gr.Column(scale=1):  # On donne plus de poids à l'upload
                            file_input = gr.File(
                                file_count="multiple",
                                label="Déposez vos fichiers ici",
                                elem_classes=["dashed-upload"],
                                scale=4,
                            )

                        with gr.Column(scale=1):  # Moins de poids pour le bouton live
                            record_name = gr.Textbox(
                                label="Nom de l'enregistrement",
                                placeholder="ex: réunion_équipe",
                                scale=2,
                            )

                            btn_record = gr.Button(
                                "🎙️ Lancer l'enregistrement live",
                                variant="primary",
                                scale=1,
                            )
                            record_status = gr.Textbox(
                                value="",
                                container=False,
                                interactive=False,
                                scale=3,
                                label="",
                            )

                    upload_btn = gr.Button("Lancer l'indexation", variant="primary")

                is_doc_rec = gr.State(False)

                with gr.Row():
                    gr.Markdown("## Documents indexés")
                    with gr.Row():  # Boutons d'action compacts en haut à droite
                        refresh_docs_btn = gr.Button(
                            "Rafraîchir", variant="secondary", min_width=50
                        )
                        classify_docs_btn = gr.Button("Classifier", variant="primary")

                # La table devient l'élément central
                doc_table = gr.Dataframe(
                    headers=["Nom", "Catégorie", "Date"],
                    value=_build_doc_table(store),
                    interactive=False,
                    wrap=True,
                )

                def handle_doc_recording(recording: bool, name: str, folder: str):
                    if not recording:
                        start_recording()
                        return (
                            True,
                            gr.update(value="⏹ Arrêter", variant="stop"),
                            "⏺ Enregistrement en cours...",
                        )
                    else:
                        _, transcript = stop_recording()
                        if not transcript or transcript.startswith("("):
                            return (
                                False,
                                gr.update(value="🎙️ Démarrer", variant="primary"),
                                "Aucun audio capturé.",
                            )

                        rec_name = name.strip() or "enregistrement"
                        txt_filename = f"{rec_name}.txt"
                        txt_path = os.path.join(tempfile.gettempdir(), txt_filename)

                        with open(txt_path, "w", encoding="utf-8") as f:
                            f.write(transcript)

                        folder_name = folder.strip() or "enregistrements"
                        msg = document_loader.save_files([txt_path], folder_name)

                        return (
                            False,
                            gr.update(value="🎙️ Démarrer", variant="primary"),
                            f"✅ {msg}",
                        )

                btn_record.click(
                    fn=handle_doc_recording,
                    inputs=[is_doc_rec, record_name, folder_input],
                    outputs=[is_doc_rec, btn_record, record_status],
                ).then(fn=lambda: _build_doc_table(store), outputs=[doc_table])

                def handle_upload_v2(files, folder_name):
                    if files is None:
                        return _build_doc_table(store)

                    processed_paths = []

                    for f in files:
                        if isinstance(f, dict):
                            file_path = f["name"]
                        else:
                            file_path = f.name

                        if file_path.lower().endswith((".mp3", ".mp4", ".wav", ".m4a")):
                            status, transcript = transcribe_file(file_path)

                            if transcript:
                                clean_name = os.path.basename(file_path)
                                name_only = os.path.splitext(clean_name)[0]
                                txt_filename = f"{name_only}.txt"
                                txt_path = os.path.join(
                                    os.path.dirname(file_path), txt_filename
                                )

                                with open(txt_path, "w", encoding="utf-8") as temp_file:
                                    temp_file.write(transcript)

                                processed_paths.append(txt_path)

                        else:
                            processed_paths.append(file_path)

                    if processed_paths:
                        msg = document_loader.save_files(processed_paths, folder_name)
                        print(f"DEBUG: DocumentLoader dit : {msg}")

                    return _build_doc_table(store)

                upload_btn.click(
                    fn=handle_upload_v2,
                    inputs=[file_input, folder_input],
                    outputs=[doc_table],
                    queue=True,
                )

                refresh_docs_btn.click(
                    fn=lambda: _build_doc_table(store),
                    outputs=[doc_table],
                )

                def handle_classify_docs():
                    run_async(agent.classify_uncategorized_docs())
                    return _build_doc_table(store)

                classify_docs_btn.click(
                    fn=handle_classify_docs,
                    outputs=[doc_table],
                )

            # ── Tab 3: Mails (tableau) ─────────────────────────────────
            with gr.Tab("Mails"):
                with gr.Row():
                    gr.Markdown("## Mails synchronisés et classifiés")
                    # Un petit bouton discret en haut à droite
                    refresh_btn = gr.Button(
                        "Rafraîchir", variant="secondary", scale=0, min_width=150
                    )

                with gr.Accordion("Mails reçus", open=True):
                    received_table = gr.Dataframe(
                        headers=["Sujet", "Expéditeur", "Catégorie", "Date"],
                        value=_build_mail_table(store, "received"),
                        interactive=False,
                        wrap=True,
                    )

                with gr.Accordion(
                    "Mails envoyés", open=False
                ):  # Fermé par défaut pour gagner de la place
                    sent_table = gr.Dataframe(
                        headers=["Sujet", "Destinataire", "Catégorie", "Date"],
                        value=_build_mail_table(store, "sent"),
                        interactive=False,
                        wrap=True,
                    )

                def refresh_tables() -> tuple[list[list[str]], list[list[str]]]:
                    gr.Info("Tableaux mis à jour")
                    return (
                        _build_mail_table(store, "received"),
                        _build_mail_table(store, "sent"),
                    )

                refresh_btn.click(
                    fn=refresh_tables,
                    outputs=[received_table, sent_table],
                )

            # ── Tab 4: Calendrier (vue visuelle) ───────────────────────
            with gr.Tab("Calendrier"):
                today = date.today()
                cal_year = gr.State(today.year)
                cal_month = gr.State(today.month)

                with gr.Row():
                    gr.Markdown("## 🗓️ Mon Agenda")
                    refresh_cal_btn = gr.Button(
                        "Rafraîchir", variant="secondary", scale=0, min_width=150
                    )

                cal_html = gr.HTML(
                    value=_build_calendar_html(),
                    label="",
                )

                with gr.Row(elem_classes=["cal-nav-row"]):
                    btn_prev = gr.Button("◀ Mois précédent", variant="secondary", scale=0, min_width=160)
                    btn_today = gr.Button("Aujourd'hui", variant="primary", scale=0, min_width=120)
                    btn_next = gr.Button("Mois suivant ▶", variant="secondary", scale=0, min_width=160)

                with gr.Accordion("Liste des événements", open=False):
                    cal_table = gr.Dataframe(
                        headers=["Date", "Heure", "Événement"],
                        value=_build_calendar_table(),
                        interactive=False,
                        wrap=True,
                    )

                def go_prev(y: int, m: int) -> tuple[str, list[list[str]], int, int]:
                    m -= 1
                    if m < 1:
                        m, y = 12, y - 1
                    return _build_calendar_html(y, m), _build_calendar_table(), y, m

                def go_next(y: int, m: int) -> tuple[str, list[list[str]], int, int]:
                    m += 1
                    if m > 12:
                        m, y = 1, y + 1
                    return _build_calendar_html(y, m), _build_calendar_table(), y, m

                def go_today() -> tuple[str, list[list[str]], int, int]:
                    t = date.today()
                    return _build_calendar_html(t.year, t.month), _build_calendar_table(), t.year, t.month

                def refresh_cal(y: int, m: int) -> tuple[str, list[list[str]]]:
                    return _build_calendar_html(y, m), _build_calendar_table()

                btn_prev.click(
                    fn=go_prev,
                    inputs=[cal_year, cal_month],
                    outputs=[cal_html, cal_table, cal_year, cal_month],
                )
                btn_next.click(
                    fn=go_next,
                    inputs=[cal_year, cal_month],
                    outputs=[cal_html, cal_table, cal_year, cal_month],
                )
                btn_today.click(
                    fn=go_today,
                    outputs=[cal_html, cal_table, cal_year, cal_month],
                )
                refresh_cal_btn.click(
                    fn=refresh_cal,
                    inputs=[cal_year, cal_month],
                    outputs=[cal_html, cal_table],
                )

    return cast(gr.Blocks, demo)
