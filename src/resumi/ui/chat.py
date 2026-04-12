"""Bridge between UI and the Agent. Manages conversation state."""

import asyncio
from collections.abc import Coroutine
from typing import Any

from resumi.core.agent import Agent


def ask(
    agent: Agent,
    message: str,
    history: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    """Synchronous wrapper around agent.chat() for Gradio callbacks."""
    return run_async(agent.chat(message=message, history=history))


def run_async[T](coro: Coroutine[Any, Any, T]) -> T:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
