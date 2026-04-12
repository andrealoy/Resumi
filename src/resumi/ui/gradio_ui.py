"""Gradio Blocks interface."""

from typing import cast

import gradio as gr

from resumi.core.agent import Agent
from resumi.ui.chat import ask
from resumi.core.mail_loader import MailLoader
from resumi.core.document_loader import DocumentLoader
def create_gradio_blocks(
    *,
    agent: Agent,
    mail_loader: MailLoader,
    document_loader: DocumentLoader,
) -> gr.Blocks:
    with gr.Blocks(title="Resumi") as demo:
        with gr.Tabs():

            with gr.Tab("Chat Assistant"):
                gr.Markdown(
                    "# Resumi\nSynchronise Gmail, puis discute avec l'assistant."
                )

                chatbot = gr.Chatbot(
                    label="Conversation",
                    type="messages",
                    allow_tags=False,
                )

                message = gr.Textbox(
                    label="Message",
                    placeholder="Pose une question…",
                )

                sources = gr.JSON(label="Sources RAG")

                def respond(
                    text: str,
                    history: list[dict[str, str]],
                ) -> tuple[
                    list[dict[str, str]],
                    str,
                    list[dict[str, object]],
                ]:
                    result = ask(agent, text)

                    updated = history + [
                        {"role": "user", "content": text},
                        {
                            "role": "assistant",
                            "content": str(result["answer"]),
                        },
                    ]

                    source_list = result.get("sources", [])

                    return updated, "", source_list  # type: ignore[return-value]

                message.submit(
                    respond,
                    inputs=[message, chatbot],
                    outputs=[chatbot, message, sources],
                )
            with gr.Tab("Upload Documents"):

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

                def handle_upload(files, folder_name):
                    return document_loader.save_files(files, folder_name)

                upload_btn.click(
                fn=handle_upload,
                inputs=[file_input, folder_input],
                outputs=upload_status,
                )
            with gr.Tab("Assistant Mail"):
                email_selector = gr.Dropdown(
                    choices=mail_loader.list_email_files(),
                    label="Choisir un mail synchronisé",
                )

                email_input = gr.Textbox(
                    label="Contenu du mail",
                    lines=10,
                    placeholder="Colle ici le contenu du mail...",
                )

                def load_selected_email(path: str) -> str:
                    if not path:
                       return ""
                    return mail_loader.read_email_file(path)

                email_selector.change(
                    fn=load_selected_email,
                    inputs=email_selector,
                    outputs=email_input,
                )

                classify_btn = gr.Button("Classifier le thème")

                category_output = gr.Textbox(
                    label="Catégorie thématique proposée"
                )

                async def classify_with_llm(email_text: str) -> str:
                    result = await agent.classify_email(email_text=email_text)
                    return result["raw_result"]

                classify_btn.click(
                    fn=classify_with_llm,
                    inputs=email_input,
                    outputs=category_output,
                )
                
                draft_btn = gr.Button("Générer un brouillon")

                draft_output = gr.Textbox(
                    label="Brouillon de réponse",
                    lines=10,
                )

                async def generate_reply(email_text: str) -> str:
                    return await agent.draft_email_reply(email_text=email_text)
 
                draft_btn.click(
                    fn=generate_reply,
                    inputs=email_input,
                    outputs=draft_output,
                )
    return cast(gr.Blocks, demo)