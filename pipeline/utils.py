"""Shared helpers: hybrid LLM routing (Ollama local + Anthropic for heavy steps),
timing decorator, config loader.

Routing strategy:
  tier="local"  → always Ollama (free, good enough for simple tasks)
  tier="fast"   → Anthropic Claude if available (faster, better reasoning),
                   falls back to Ollama if no API key
"""

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

_init_done = False
_ollama_ok = False
_anthropic_ok = False
_anthropic_client = None


def _check_ollama() -> bool:
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3):
            return True
    except Exception:
        return False


def _check_anthropic() -> bool:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    return bool(key and key.strip())


def _init_backends():
    """One-time init: load .env, probe both backends."""
    global _init_done, _ollama_ok, _anthropic_ok
    if _init_done:
        return
    _init_done = True

    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

    _ollama_ok = _check_ollama()
    _anthropic_ok = _check_anthropic()

    backends = []
    if _ollama_ok:
        backends.append("Ollama (local)")
    if _anthropic_ok:
        backends.append("Anthropic (cloud — for heavy steps)")
    if not backends:
        backends.append("DRY-RUN (no backends available)")
    print(f"  [utils] Backends: {' + '.join(backends)}")


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.Anthropic()
    return _anthropic_client


def load_config() -> dict[str, Any]:
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f) or {}


# ── Ollama ──────────────────────────────────────────────────────

def _ask_ollama(prompt: str, *, system: str, model: str, temperature: float) -> str:
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
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


# ── Anthropic ───────────────────────────────────────────────────

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


# ── Router ──────────────────────────────────────────────────────

def _resolve_backend(tier: str) -> str:
    """Pick backend for this call.

    tier="local"  → Ollama (always, unless down → Anthropic → dry_run)
    tier="fast"   → Anthropic if available, else Ollama, else dry_run
    """
    _init_backends()

    if tier == "fast":
        if _anthropic_ok:
            return "anthropic"
        if _ollama_ok:
            return "ollama"
        return "dry_run"
    else:  # "local"
        if _ollama_ok:
            return "ollama"
        if _anthropic_ok:
            return "anthropic"
        return "dry_run"


def _get_model(backend: str) -> str:
    cfg = load_config().get("pipeline", {})
    if backend == "anthropic":
        return cfg.get("anthropic_model", "claude-sonnet-4-20250514")
    return cfg.get("ollama_model", "llama3.1:8b")


# ── Public API ──────────────────────────────────────────────────

def ask_llm(
    prompt: str,
    *,
    system: str = "",
    model: str | None = None,
    temperature: float = 0.3,
    tier: str = "local",
    dry_run_response: str = "",
) -> str:
    """Send a prompt to an LLM.

    tier="local"  → Ollama handles it (free, for lightweight tasks)
    tier="fast"   → Anthropic Claude if key is set (for heavy/slow steps)
    """
    backend = _resolve_backend(tier)

    if backend == "dry_run":
        return dry_run_response or '{"stub": true}'

    resolved_model = model or _get_model(backend)

    if backend == "anthropic":
        return _ask_anthropic(prompt, system=system, model=resolved_model,
                              temperature=temperature)
    return _ask_ollama(prompt, system=system, model=resolved_model,
                       temperature=temperature)


def ask_llm_json(
    prompt: str,
    *,
    system: str = "",
    model: str | None = None,
    tier: str = "local",
    dry_run_response: dict | list | None = None,
) -> Any:
    """Send a prompt requesting JSON, parse and return.

    tier="local" / tier="fast" — same routing as ask_llm.
    """
    backend = _resolve_backend(tier)

    if backend == "dry_run":
        return dry_run_response if dry_run_response is not None else {"stub": True}

    raw = ask_llm(
        prompt,
        system=(system + "\nRespond ONLY with valid JSON. No markdown fences, "
                "no explanation, no text before or after the JSON."),
        model=model,
        tier=tier,
    )

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        first_newline = cleaned.find("\n")
        if first_newline > 0:
            cleaned = cleaned[first_newline + 1:]
        if cleaned.endswith("```"):
            cleaned = cleaned[: cleaned.rfind("```")]
        cleaned = cleaned.strip()

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
