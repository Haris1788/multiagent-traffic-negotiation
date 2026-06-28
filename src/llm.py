"""Vendor-neutral text-generation backend.

A single entry point, :func:`complete`, that every agent uses to get text from a
language model. It calls an OpenAI-compatible ``/chat/completions`` endpoint over HTTP,
so it works with a **free local server** (Ollama, LM Studio, llama.cpp, vLLM, ...) or
any compatible endpoint — no vendor lock-in and no required API key.

Swapping the model/provider is a one-line change in ``.env`` (see ``src/config.py``);
no agent code changes.
"""
from __future__ import annotations

from functools import lru_cache

import httpx

from src import config


def complete(prompt: str, *, model: str | None = None, system: str | None = None,
             max_tokens: int = 1024, timeout: float | None = None) -> str:
    """Send a prompt to the configured LLM backend and return the text reply."""
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    headers = {}
    if config.LLM_API_KEY:
        headers["Authorization"] = f"Bearer {config.LLM_API_KEY}"

    resp = httpx.post(
        f"{config.LLM_BASE_URL.rstrip('/')}/chat/completions",
        json={
            "model": model or config.LLM_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
        },
        headers=headers,
        timeout=timeout or config.LLM_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


@lru_cache(maxsize=1)
def is_available(timeout: float = 0.5) -> bool:
    """Return whether the configured OpenAI-compatible backend responds quickly."""
    try:
        resp = httpx.get(
            f"{config.LLM_BASE_URL.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {config.LLM_API_KEY}"} if config.LLM_API_KEY else {},
            timeout=timeout,
        )
        return resp.status_code < 500
    except Exception:
        return False
