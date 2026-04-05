"""Shared helpers: Ollama local LLM wrapper, timing decorator, config loader."""

from __future__ import annotations

import functools
import json
import os
import time
import urllib.request
import urllib.error
from typing import Any

import yaml


OLLAMA_BASE = "http://localhost:11434"


def _ollama_available() -> bool:
    """Return True if Ollama server is reachable."""
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3):
            return True
    except Exception:
        return False


def load_config() -> dict[str, Any]:
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f) or {}


def ask_llm(
    prompt: str,
    *,
    system: str = "",
    model: str | None = None,
    temperature: float = 0.3,
    dry_run_response: str = "",
) -> str:
    """Send a prompt to the local Ollama LLM and return the text response.

    Falls back to *dry_run_response* when Ollama isn't running.
    """
    if not _ollama_available():
        print("  [utils] Ollama not reachable — returning dry-run response")
        return dry_run_response or '{"stub": true}'

    cfg = load_config()
    model = model or cfg.get("pipeline", {}).get("model", "llama3.1:8b")

    full_prompt = prompt
    if system:
        full_prompt = f"{system}\n\n{prompt}"

    payload = json.dumps({
        "model": model,
        "prompt": full_prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode())

    return data.get("response", "")


def ask_llm_json(
    prompt: str,
    *,
    system: str = "",
    model: str | None = None,
    dry_run_response: dict | list | None = None,
) -> Any:
    """Send a prompt to Ollama requesting JSON output, parse and return."""
    if not _ollama_available():
        print("  [utils] Ollama not reachable — returning dry-run response")
        return dry_run_response if dry_run_response is not None else {"stub": True}

    raw = ask_llm(
        prompt,
        system=(system + "\nRespond ONLY with valid JSON. No markdown fences, "
                "no explanation, no text before or after the JSON."),
        model=model,
    )

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        first_newline = cleaned.find("\n")
        if first_newline > 0:
            cleaned = cleaned[first_newline + 1:]
        if cleaned.endswith("```"):
            cleaned = cleaned[: cleaned.rfind("```")]
        cleaned = cleaned.strip()

    # Find the JSON object/array boundary
    start = -1
    for i, ch in enumerate(cleaned):
        if ch in ("{", "["):
            start = i
            break
    if start > 0:
        cleaned = cleaned[start:]

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        if dry_run_response is not None:
            print(f"  [utils] JSON parse failed, using fallback. Raw: {cleaned[:200]}")
            return dry_run_response
        raise


def timed(step_name: str):
    """Decorator that records wall-clock seconds for a pipeline step."""

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(state, *args, **kwargs):
            t0 = time.time()
            result = fn(state, *args, **kwargs)
            elapsed = round(time.time() - t0, 1)
            print(f"  [{step_name}] {elapsed}s")
            if isinstance(result, dict):
                timing = result.get("timing", {})
                timing[step_name] = elapsed
                result["timing"] = timing
            return result

        return wrapper

    return decorator
