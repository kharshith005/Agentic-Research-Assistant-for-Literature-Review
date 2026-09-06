"""Shared Gemini chat model for all agents."""

from __future__ import annotations

import re
import time
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI

MODEL = "gemini-flash-lite-latest"
MAX_RETRIES = 5


def get_llm(temperature: float = 0) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model=MODEL, temperature=temperature)


def _retry_seconds(exc: Exception) -> float:
    match = re.search(r"retry in ([\d.]+)s", str(exc), re.IGNORECASE)
    if match:
        return min(float(match.group(1)) + 1.0, 60.0)
    return 5.0


def invoke_llm(llm: ChatGoogleGenerativeAI, messages: list[BaseMessage]) -> Any:
    """Invoke the chat model with backoff on transient Gemini errors."""
    last_error: Exception | None = None
    # Disable AFC: we don't use tools, and it triggers an SDK warning via LangChain.
    invoke_kwargs = {"automatic_function_calling": {"disable": True}}
    for attempt in range(MAX_RETRIES):
        try:
            return llm.invoke(messages, **invoke_kwargs)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            message = str(exc).lower()
            transient = any(
                token in message
                for token in ("503", "unavailable", "429", "resource_exhausted", "high demand")
            )
            if not transient:
                raise
            time.sleep(_retry_seconds(exc) if attempt == 0 else min(2 ** attempt * 2, 45))
    assert last_error is not None
    raise last_error
