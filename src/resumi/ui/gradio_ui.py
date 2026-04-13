"""Gradio Blocks interface."""

from collections.abc import Generator
from typing import cast

import gradio as gr

from resumi.core.agent import Agent
from resumi.core.audio_handler import start_recording, stop_recording, transcribe_file
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

# --- Thème de l'interface 
theme = gr.themes.Soft(
    # Le fond sera géré via neutral_hue
    primary_hue="gray",  
    secondary_hue="blue", 
    neutral_hue="slate", 
    spacing_size="md",
    radius_size="lg",
).set(
    body_background_fill_dark="#1E293B",
    block_background_fill_dark="#334155",

    # --- BOUTONS ---
    button_primary_background_fill="#A16454", 
    button_primary_background_fill_hover="#A16454",
    button_primary_text_color="#FFFFFF",
    
    # Largeur et espacement
    layout_gap="12px",
    container_radius="12px",
)

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
    with gr.Blocks(
    title="Resumi", 
    theme=theme, 
    css=""
    ) as demo:
        gr.Markdown(
                    "# Resumi\nSynchroniser vos mails, synthétiser un fichier audio ou classer des documents."
                )
        with gr.Tabs():
            # ── Tab 1: Chat Assistant ──────────────────────────────────
            with gr.Tab("Chat"):
                
                connected = gmail_handler.is_connected()

                # --- message de bienvenue
                initial_msgs = [
                    {"role": "assistant", "content": "Bonjour ! Que voulez-vous faire aujourd'hui ?"}
                ]

                # --- chatbox
                chatbot = gr.Chatbot(
                    type="messages",
                    allow_tags=False,
                    value=initial_msgs,
                    label="",
                    show_label=False
                )

                # --- message input + audio buttons 
                with gr.Row():
                    message = gr.Textbox(
                        placeholder="Demander à Resumi",
                        scale=6,
                        lines=1,
                        container=False,
                    )
                    btn_mic = gr.Button("🎙️", variant="secondary", scale=1, min_width=120)
                    btn_stop = gr.Button("🛑", variant="stop", scale=1, min_width=80)

                # --- connexion to gmail
                with gr.Row():
                    gmail_connect_btn = gr.Button(
                        "Se connecter", variant="primary", scale=1, min_width=150,
                    )

                    gmail_status = gr.Textbox(
                        value="Gmail : connecté" if connected else "Gmail : non connecté",
                        container=False, interactive=False, scale=6, lines=1,
                    )


                # --- others
                with gr.Accordion("Sources consultées", open=False):
                    sources = gr.JSON(label="Sources RAG", show_label=False)

                is_rec  = gr.State(False)

                # --- functions connexion to gmail

                def handle_gmail_btn(history: list[dict[str, str]]):
                    if not gmail_handler.is_connected():
                        ok = gmail_handler.connect()
                        if not ok:
                            yield history, "La connexion Gmail a échoué. Vérifie que le fichier client-secret est présent dans credentials/ et réessaie.", gr.update(value="Connexion")
                            return
                        
                        yield history, "Gmail : connecté", gr.update(value="Synchronisation", variant="secondary")

                    # Lancement de la synchro
                    for new_history in _run_sync_steps(gmail_handler, agent, history):
                        yield new_history, "Synchronisation en cours...", gr.update()

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
                ) -> Generator[tuple[list[dict[str, str]], str], None, None]:
                    if not needs_auto_sync:
                        yield history, "Gmail : connecté"
                        return
                    for history, status in handle_gmail_btn(history):
                        yield history, status
 
                #demo.load(fn=auto_sync_on_load, inputs=[chatbot], outputs=[chatbot, gmail_status])

                # --- functions responses from agents
                def respond(
                    text: str,
                    history: list[dict[str, str]],
                ) -> tuple[list[dict[str, str]], str, list[dict[str, object]]]:
                    result = ask(agent, text, history)
                    updated = history + [
                        {"role": "user", "content": text},
                        {"role": "assistant", "content": str(result["answer"])},
                    ]
                    return updated, "", result.get("sources", [])  # type: ignore[return-value]
 
                # --- functions mic

                def stop_and_transcribe(history: list[dict[str, str]]):
                    _, transcript = stop_recording()
                    
                    if transcript and not transcript.startswith("("):
                        new_history = history + [
                            {"role": "user", "content": transcript},
                            {"role": "assistant", "content": "*En train de réfléchir...*"}
                        ]
                        return False, gr.update(value="🎙️", variant="secondary"), new_history, transcript
                    
                    return False, gr.update(value="🎙️", variant="secondary"), history, ""

                def agent_voice_respond(transcript: str, history: list[dict[str, str]]):
                    if not transcript or transcript.startswith("("):
                        return history, ""

                    result = ask(agent, transcript, history[:-2]) # On enlève les 2 derniers (user + loading) pour l'appel
                    
                    new_history = history[:-1] + [
                        {"role": "assistant", "content": str(result["answer"])}
                    ]
                    return new_history, result.get("sources", [])


                '''def toggle_mic(recording: bool, history: list[dict[str, str]]):
                    if not recording:
                        # Start recording
                        start_recording()
                        return (
                            True,
                            gr.update(value="⏹", variant="stop"),
                            history,
                            "",
                        )
                    else:
                        # Stop and transcribe
                        _, transcript = stop_recording()
                        new_history = history
                        if transcript and not transcript.startswith("("):
                            result = ask(agent, transcript, history)
                            new_history = history + [
                                {"role": "user",      "content": f"{transcript}"},
                                {"role": "assistant", "content": str(result["answer"])},
                            ]
                        return (
                            False,
                            gr.update(value="▶", variant="secondary"),
                            new_history,
                            "",
                        )'''
 
                # Un état caché pour stocker le texte transcrit entre les deux étapes
                temp_transcript = gr.State("")

                chat_event = message.submit(respond, inputs=[message, chatbot], outputs=[chatbot, message, sources])
               
                voice_event = btn_mic.click(
                    fn=lambda r: (True, gr.update(value="⏹", variant="stop")) if not r else (False, gr.update(value="⏳", interactive=False)),
                    inputs=[is_rec],
                    outputs=[is_rec, btn_mic],
                    show_progress=False
                ).then(
                    fn=lambda r, h: start_recording() if r else None,
                    inputs=[is_rec, chatbot],
                ).then(
                    fn=lambda r, h: stop_and_transcribe(h) if not r else (True, gr.update(), h, ""),
                    inputs=[is_rec, chatbot],
                    outputs=[is_rec, btn_mic, chatbot, temp_transcript],
                    queue=True
                ).then(
                    fn=agent_voice_respond,
                    inputs=[temp_transcript, chatbot],
                    outputs=[chatbot, sources],
                    queue=True
                )

                btn_stop.click(
                    fn=lambda: (False, gr.update(value="🎙️", variant="secondary", interactive=True), gr.update(visible=False)),
                    outputs=[is_rec, btn_mic, btn_stop],
                    cancels=[chat_event, voice_event]
                )

            # ── Tab 2: Upload Documents ────────────────────────────────
            with gr.Tab("Documents"):
                with gr.Row():
                    gr.Markdown("## Documents indexés")
                    with gr.Row(): # Boutons d'action compacts en haut à droite
                        refresh_docs_btn = gr.Button("Rafraîchir", variant="secondary", min_width=50)
                        classify_docs_btn = gr.Button("Classifier", variant="primary")

                # La table devient l'élément central
                doc_table = gr.Dataframe(
                    headers=["Nom", "Catégorie", "Date"],
                    value=_build_doc_table(store),
                    interactive=False,
                    wrap=True,
                )

                # Zone d'ajout rétractable pour ne pas encombrer l'écran
                with gr.Accordion("Ajouter de nouveaux documents", open=False):
                    with gr.Row():
                        folder_input = gr.Textbox(
                            label="Dossier cible",
                            placeholder="ex: factures_2024",
                            scale=2
                        )
                        file_input = gr.File(
                            file_count="multiple",
                            label="Déposez vos fichiers ici",
                            scale=4
                        )
                    upload_btn = gr.Button("Lancer l'indexation", variant="primary")

                # Suppression du gr.Textbox "Statut" au profit de gr.Info dans la fonction
                def handle_upload_v2(files, folder_name):
                    msg = document_loader.save_files(files, folder_name)
                    gr.Info(msg) # Notification flottante au lieu d'un bloc fixe
                    return _build_doc_table(store)

                upload_btn.click(fn=handle_upload_v2, inputs=[file_input, folder_input], outputs=[doc_table])

            # ── Tab 3: Mails (tableau) ─────────────────────────────────
            with gr.Tab("Mails"):

                with gr.Row():
                    gr.Markdown("## Mails synchronisés et classifiés")
                    # Un petit bouton discret en haut à droite
                    refresh_btn = gr.Button("Rafraîchir", variant="secondary", scale=0, min_width=150)

                with gr.Accordion("Mails reçus", open=True):
                    received_table = gr.Dataframe(
                        headers=["Sujet", "Expéditeur", "Catégorie", "Date"],
                        value=_build_mail_table(store, "received"),
                        interactive=False,
                        wrap=True
                    )

                with gr.Accordion("Mails envoyés", open=False): # Fermé par défaut pour gagner de la place
                    sent_table = gr.Dataframe(
                        headers=["Sujet", "Destinataire", "Catégorie", "Date"],
                        value=_build_mail_table(store, "sent"),
                        interactive=False,
                        wrap=True
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

    return cast(gr.Blocks, demo)
