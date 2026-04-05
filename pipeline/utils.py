"""Shared helpers: LLM wrapper (Ollama local → Anthropic cloud fallback),
timing decorator, config loader."""

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

_backend: str | None = None  # "ollama", "anthropic", or "dry_run"
_anthropic_client = None


def _ollama_available() -> bool:
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3):
            return True
    except Exception:
        return False


def _anthropic_available() -> bool:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    return bool(key and key.strip())


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.Anthropic()
    return _anthropic_client


def _detect_backend() -> str:
    """Auto-detect which LLM backend to use. Priority: Ollama > Anthropic > dry-run."""
    global _backend
    if _backend is not None:
        return _backend

    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

    if _ollama_available():
        _backend = "ollama"
        print("  [utils] Backend: Ollama (local)")
    elif _anthropic_available():
        _backend = "anthropic"
        print("  [utils] Backend: Anthropic Claude (cloud)")
    else:
        _backend = "dry_run"
        print("  [utils] Backend: DRY-RUN (no Ollama or Anthropic key found)")

    return _backend


def load_config() -> dict[str, Any]:
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f) or {}


# ── Ollama backend ──────────────────────────────────────────────

def _ask_ollama(prompt: str, *, system: str, model: str, temperature: float) -> str:
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


# ── Anthropic backend ───────────────────────────────────────────

def _ask_anthropic(prompt: str, *, system: str, model: str, temperature: float) -> str:
    client = _get_anthropic_client()
    kwargs: dict[str, Any] = dict(
        model=model,
        max_tokens=4096,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    if system:
        kwargs["system"] = system
    resp = client.messages.create(**kwargs)
    return resp.content[0].text


# ── Public API ──────────────────────────────────────────────────

def ask_llm(
    prompt: str,
    *,
    system: str = "",
    model: str | None = None,
    temperature: float = 0.3,
    dry_run_response: str = "",
) -> str:
    """Send a prompt to the best available LLM and return text.

    Auto-detection order: Ollama (local) → Anthropic (cloud) → dry-run stub.
    """
    backend = _detect_backend()

    if backend == "dry_run":
        return dry_run_response or '{"stub": true}'

    cfg = load_config()
    pipeline_cfg = cfg.get("pipeline", {})

    if backend == "ollama":
        resolved_model = model or pipeline_cfg.get("ollama_model",
                                                    pipeline_cfg.get("model", "llama3.1:8b"))
        return _ask_ollama(prompt, system=system, model=resolved_model,
                           temperature=temperature)

    # anthropic
    resolved_model = model or pipeline_cfg.get("anthropic_model", "claude-sonnet-4-20250514")
    return _ask_anthropic(prompt, system=system, model=resolved_model,
                          temperature=temperature)


def ask_llm_json(
    prompt: str,
    *,
    system: str = "",
    model: str | None = None,
    dry_run_response: dict | list | None = None,
) -> Any:
    """Send a prompt requesting JSON output, parse and return."""
    backend = _detect_backend()

    if backend == "dry_run":
        return dry_run_response if dry_run_response is not None else {"stub": True}

    raw = ask_llm(
        prompt,
        system=(system + "\nRespond ONLY with valid JSON. No markdown fences, "
                "no explanation, no text before or after the JSON."),
        model=model,
    )

    cleaned = raw.strip()
    # Strip markdown fences
    if cleaned.startswith("```"):
        first_newline = cleaned.find("\n")
        if first_newline > 0:
            cleaned = cleaned[first_newline + 1:]
        if cleaned.endswith("```"):
            cleaned = cleaned[: cleaned.rfind("```")]
        cleaned = cleaned.strip()

    # Find JSON start
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
