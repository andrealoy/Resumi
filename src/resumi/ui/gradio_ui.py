"""Gradio Blocks interface."""

from typing import cast

import gradio as gr

from resumi.core.agent import Agent
from resumi.ui.chat import ask


def create_gradio_blocks(*, agent: Agent) -> gr.Blocks:
    with gr.Blocks(title="Resumi") as demo:
        gr.Markdown("# Resumi\nSynchronise Gmail, puis discute avec l'assistant.")
        chatbot = gr.Chatbot(label="Conversation", type="messages", allow_tags=False)
        message = gr.Textbox(label="Message", placeholder="Pose une question…")
        sources = gr.JSON(label="Sources RAG")

        def respond(
            text: str,
            history: list[dict[str, str]],
        ) -> tuple[list[dict[str, str]], str, list[dict[str, object]]]:
            result = ask(agent, text)
            updated = history + [
                {"role": "user", "content": text},
                {"role": "assistant", "content": str(result["answer"])},
            ]
            source_list = result.get("sources", [])
            return updated, "", source_list  # type: ignore[return-value]

        message.submit(
            respond, inputs=[message, chatbot], outputs=[chatbot, message, sources]
        )

    return cast(gr.Blocks, demo)
