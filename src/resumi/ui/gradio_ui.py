"""Gradio Blocks interface."""

from collections.abc import Generator
from typing import cast

import gradio as gr



from resumi.core.agent import Agent
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


def _initial_messages(gmail_handler: GmailHandler) -> list[dict[str, str]]:
    """Build startup assistant messages depending on Gmail state."""
    msgs: list[dict[str, str]] = []

    if not gmail_handler.is_connected():
        msgs.append(
            {
                "role": "assistant",
                "content": (
                    "Bonjour ! Gmail n'est pas encore connecté.\n\n"
                    "Clique sur **Connecter Gmail** ci-dessous pour "
                    "autoriser l'accès à ta boîte mail."
                ),
            }
        )
        return msgs

    age_label = _sync_age_label(gmail_handler)
    age = gmail_handler.sync_age_hours()

    if age is None or age >= _STALE_HOURS:
        msgs.append(
            {
                "role": "assistant",
                "content": (
                    f"Bonjour ! Gmail est connecté. {age_label}\n\n"
                    "Les mails ne sont pas à jour "
                    "— synchronisation automatique en cours…"
                ),
            }
        )
    else:
        msgs.append(
            {
                "role": "assistant",
                "content": (
                    f"Bonjour ! Gmail est connecté. {age_label}\n\n"
                    "Tu peux poser des questions, demander un brouillon de réponse, "
                    "ou relancer une synchronisation."
                ),
            }
        )
    return msgs


def _run_sync_steps(
    gmail_handler: GmailHandler,
    agent: Agent,
    history: list[dict[str, str]],
) -> Generator[list[dict[str, str]], None, None]:
    """Generator that yields chat history updates during a step-by-step Gmail sync."""

    def _msg(text: str) -> dict[str, str]:
        return {"role": "assistant", "content": text}

    # Step 1 — fetch received
    history = history + [_msg("📥 Récupération des mails reçus…")]
    yield history
    try:
        received = run_async(gmail_handler.fetch_received())
    except Exception as exc:
        yield history + [_msg(f"❌ Erreur récupération mails reçus : {exc}")]
        return
    history = history + [_msg(f"✅ **{len(received)}** mails reçus récupérés.")]
    yield history

    # Step 2 — fetch sent
    history = history + [_msg("📤 Récupération des mails envoyés…")]
    yield history
    try:
        sent = run_async(gmail_handler.fetch_sent())
    except Exception as exc:
        yield history + [_msg(f"❌ Erreur récupération mails envoyés : {exc}")]
        return
    history = history + [_msg(f"✅ **{len(sent)}** mails envoyés récupérés.")]
    yield history

    # Step 3 — save to disk
    history = history + [_msg("💾 Sauvegarde des mails sur le disque…")]
    yield history
    saved_received = gmail_handler.save_messages(received, "received")
    saved_sent = gmail_handler.save_messages(sent, "sent")
    total_saved = len(saved_received) + len(saved_sent)
    history = history + [_msg(f"✅ **{total_saved}** mails sauvegardés.")]
    yield history

    # Step 4 — reindex
    history = history + [_msg("🔍 Indexation des documents…")]
    yield history
    indexed = gmail_handler.reindex()
    history = history + [_msg(f"✅ **{indexed}** documents indexés.")]
    yield history

    # Step 5 — classify uncategorized mails
    history = history + [_msg("🏷️ Classification automatique des mails…")]
    yield history
    try:
        classified = run_async(agent.classify_uncategorized_mails())
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
        title = str(d.get("title", ""))
        category = str(d.get("category", "")) or "—"
        date = str(d.get("date", ""))[:10]
        rows.append([title, category, date])
    return rows

def _build_calendar_table() -> list[list[str]]:
    """Lit le calendrier local CSV pour l'affichage Gradio."""
    import os
    import pandas as pd
    file_path = "local_calendar.csv"
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            return df.values.tolist()
        except:
            return []
    return []


def create_gradio_blocks(
    *,
    agent: Agent,
    mail_loader: MailLoader,
    document_loader: DocumentLoader,
    gmail_handler: GmailHandler,
    store: DocumentStore,
) -> gr.Blocks:
    with gr.Blocks(title="Resumi") as demo:
        with gr.Tabs():
            # ── Tab 1: Chat Assistant ──────────────────────────────────
            with gr.Tab("Chat"):
                gr.Markdown(
                    "# Resumi\nSynchronise Gmail, puis discute avec l'assistant.\n"
                    "💡 *Demande un brouillon de réponse "
                    "à un mail directement ici.*"
                )

                initial_msgs = _initial_messages(gmail_handler)
                connected = gmail_handler.is_connected()

                with gr.Row():
                    with gr.Column(scale=2): # Colonne Chat
                        chatbot = gr.Chatbot(
                            label="Conversation",
                            type="messages",
                            allow_tags=False,
                            value=initial_msgs,
                        )
                    
                    with gr.Column(scale=1, variant="panel"): # 'variant="panel"' donne le fond gris/encadré
                        # On utilise du HTML pour centrer le titre proprement
                        gr.HTML("<h3 style='text-align: center; margin-bottom: 10px;'>🗓️ Mon Agenda</h3>")
                        
                        calendar_table = gr.Dataframe(
                            headers=["Date", "Heure", "Événement"],
                            value=_build_calendar_table(),
                            interactive=False,
                            type="pandas",
                        )

                with gr.Row():
                    gmail_connect_btn = gr.Button(
                        "🔗 Connecter Gmail",
                        visible=not connected,
                        variant="primary",
                    )
                    gmail_sync_btn = gr.Button(
                        "🔄 Synchroniser Gmail",
                        visible=connected,
                        variant="secondary",
                    )

                gmail_status = gr.Textbox(
                    label="Statut Gmail",
                    visible=False,
                    interactive=False,
                )

                def handle_gmail_connect(
                    history: list[dict[str, str]],
                ) -> Generator[
                    tuple[list[dict[str, str]], dict, dict, dict],
                    None,
                    None,
                ]:
                    ok = gmail_handler.connect()
                    if not ok:
                        updated = history + [
                            {
                                "role": "assistant",
                                "content": (
                                    "La connexion Gmail a échoué. "
                                    "Vérifie que le fichier "
                                    "client-secret est présent "
                                    "dans credentials/ et réessaie."
                                ),
                            }
                        ]
                        yield (
                            updated,
                            gr.update(visible=True),
                            gr.update(visible=False),
                            gr.update(visible=False),
                        )
                        return

                    connected_msg = history + [
                        {
                            "role": "assistant",
                            "content": (
                                "Gmail connecté avec succès ! "
                                "Synchronisation automatique…"
                            ),
                        }
                    ]
                    yield (
                        connected_msg,
                        gr.update(visible=False),
                        gr.update(visible=True),
                        gr.update(visible=False),
                    )

                    # Auto-sync right after connecting
                    for h in _run_sync_steps(gmail_handler, agent, connected_msg):
                        yield (
                            h,
                            gr.update(visible=False),
                            gr.update(visible=True),
                            gr.update(visible=False),
                        )

                gmail_connect_btn.click(
                    fn=handle_gmail_connect,
                    inputs=[chatbot],
                    outputs=[chatbot, gmail_connect_btn, gmail_sync_btn, gmail_status],
                )

                def handle_gmail_sync(
                    history: list[dict[str, str]],
                ) -> Generator[list[dict[str, str]], None, None]:
                    yield from _run_sync_steps(gmail_handler, agent, history)

                gmail_sync_btn.click(
                    fn=handle_gmail_sync,
                    inputs=[chatbot],
                    outputs=[chatbot],
                )

                # Auto-sync on page load if mails are stale
                needs_auto_sync = connected and (
                    gmail_handler.sync_age_hours() is None
                    or (gmail_handler.sync_age_hours() or 0) >= _STALE_HOURS
                )

                def auto_sync_on_load(
                    history: list[dict[str, str]],
                ) -> Generator[list[dict[str, str]], None, None]:
                    if not needs_auto_sync:
                        yield history
                        return
                    yield from _run_sync_steps(gmail_handler, agent, history)

                demo.load(
                    fn=auto_sync_on_load,
                    inputs=[chatbot],
                    outputs=[chatbot],
                )

                message = gr.Textbox(
                    label="Message",
                    placeholder="Pose une question ou demande un brouillon de réponse…",
                )

                sources = gr.JSON(label="Sources RAG")

                def respond(
                    text: str,
                    history: list[dict[str, str]],
                ) -> tuple[
                    list[dict[str, str]],
                    str,
                    list[dict[str, object]],
                    list[list[str]], 
                ]:
                    result = ask(agent, text, history)

                    updated = history + [
                        {"role": "user", "content": text},
                        {
                            "role": "assistant",
                            "content": str(result["answer"]),
                        },
                    ]

                    source_list = result.get("sources", [])
                    
                    # On renvoie aussi le contenu du CSV mis à jour
                    return updated, "", source_list, _build_calendar_table() 

                message.submit(
                    respond,
                    inputs=[message, chatbot],
                    outputs=[chatbot, message, sources, calendar_table], # Ajout ici
                )

            # ── Tab 2: Upload Documents ────────────────────────────────
            with gr.Tab("Documents"):
                gr.Markdown("## Documents uploadés")

                doc_headers = ["Nom", "Catégorie", "Date"]
                doc_table = gr.Dataframe(
                    headers=doc_headers,
                    value=_build_doc_table(store),
                    interactive=False,
                    wrap=True,
                )

                with gr.Row():
                    classify_docs_btn = gr.Button(
                        "🏷️ Classifier les documents",
                        variant="secondary",
                    )
                    refresh_docs_btn = gr.Button(
                        "🔄 Rafraîchir",
                        variant="secondary",
                    )

                classify_status = gr.Textbox(
                    label="Statut classification",
                    interactive=False,
                )

                def classify_docs() -> tuple[str, list[list[str]]]:
                    n = run_async(agent.classify_uncategorized_docs())
                    return (
                        f"✅ {n} document(s) classifié(s).",
                        _build_doc_table(store),
                    )

                classify_docs_btn.click(
                    fn=classify_docs,
                    outputs=[classify_status, doc_table],
                )

                refresh_docs_btn.click(
                    fn=lambda: _build_doc_table(store),
                    outputs=[doc_table],
                )

                gr.Markdown("---\n### Ajouter des fichiers")

                folder_input = gr.Textbox(
                    label="Nom du dossier",
                    placeholder="ex: cours_ml",
                )

                file_input = gr.File(
                    file_count="multiple",
                    label="Ajouter des fichiers",
                )

                upload_btn = gr.Button("Uploader et indexer")

                upload_status = gr.Textbox(label="Statut")

                def handle_upload(files, folder_name) -> tuple[str, list[list[str]]]:
                    msg = document_loader.save_files(files, folder_name)
                    return msg, _build_doc_table(store)

                upload_btn.click(
                    fn=handle_upload,
                    inputs=[file_input, folder_input],
                    outputs=[upload_status, doc_table],
                )

            # ── Tab 3: Mails (tableau) ─────────────────────────────────
            with gr.Tab("Mails"):
                gr.Markdown("## Mails synchronisés et classifiés")

                received_headers = ["Sujet", "Expéditeur", "Catégorie", "Date"]
                sent_headers = ["Sujet", "Destinataire", "Catégorie", "Date"]

                gr.Markdown("### 📥 Reçus")
                received_table = gr.Dataframe(
                    headers=received_headers,
                    value=_build_mail_table(store, "received"),
                    interactive=False,
                    wrap=True,
                )

                gr.Markdown("### 📤 Envoyés")
                sent_table = gr.Dataframe(
                    headers=sent_headers,
                    value=_build_mail_table(store, "sent"),
                    interactive=False,
                    wrap=True,
                )

                refresh_btn = gr.Button("🔄 Rafraîchir")

                def refresh_tables() -> tuple[list[list[str]], list[list[str]]]:
                    return (
                        _build_mail_table(store, "received"),
                        _build_mail_table(store, "sent"),
                    )

                refresh_btn.click(
                    fn=refresh_tables,
                    outputs=[received_table, sent_table],
                )

    return cast(gr.Blocks, demo)
