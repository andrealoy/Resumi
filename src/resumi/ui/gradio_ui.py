"""Gradio Blocks interface."""

from typing import cast

import gradio as gr

from resumi.core.agent import Agent
from resumi.ui.chat import ask


def create_gradio_blocks(*, agent: Agent) -> gr.Blocks:
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

            with gr.Tab("Assistant Mail"):
                email_input = gr.Textbox(
                    label="Contenu du mail",
                    lines=10,
                    placeholder="Colle ici le contenu du mail...",
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

    return cast(gr.Blocks, demo)