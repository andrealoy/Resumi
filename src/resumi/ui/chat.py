"""Bridge between UI and the Agent. Manages conversation state."""

import asyncio
import threading
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
        # No loop running – safe to use asyncio.run()
        return asyncio.run(coro)
    # A loop is already running (FastAPI / Gradio). Run the coroutine in a
    # dedicated thread with its own event loop to avoid conflicts.
    result: T | None = None
    exc: BaseException | None = None

    def _target() -> None:
        nonlocal result, exc
        try:
            result = asyncio.run(coro)
        except BaseException as e:
            exc = e

    t = threading.Thread(target=_target)
    t.start()
    t.join()
    if exc is not None:
        raise exc
    return result  # type: ignore[return-value]
