"""Central configuration: the LLM backend + environment loading.

The project runs on a FREE, vendor-neutral language-model backend: any server that
speaks the common OpenAI-compatible ``/chat/completions`` API. By default it points at
a local model server (e.g. Ollama), so there is **no API key and no cost**. Override
the env vars in ``.env`` to use any other endpoint/model you have.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()  # reads .env if present

# Base URL of an OpenAI-compatible endpoint. Default = local server (free, no key).
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")

# Model the backend serves — any open model your server has available.
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1")

# API key: leave blank for local servers; only set if your endpoint requires one.
LLM_API_KEY = os.getenv("LLM_API_KEY", "")

# Keep local development and tests responsive when no model server is running.
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "2.0"))
