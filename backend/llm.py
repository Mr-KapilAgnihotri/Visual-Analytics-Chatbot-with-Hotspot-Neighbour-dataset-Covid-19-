"""
Thin wrapper around the LLM (Groq by default).

Groq exposes an OpenAI-COMPATIBLE API, so we use the official `openai` Python
client pointed at Groq's URL. Switching provider (a local Ollama model, OpenAI,
another host) means changing LLM_BASE_URL / LLM_MODEL in .env - no code change.

Design choices that protect the free-tier quota (8,000 tokens/minute):
  * an in-memory cache: an identical prompt never costs tokens twice
  * low reasoning effort and a small max_tokens budget
  * every failure (no key, no network, rate limit, bad JSON) returns None and
    the caller falls back to deterministic rules, so the chatbot never goes silent
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time

from backend import config

log = logging.getLogger(__name__)


class LLM:
    def __init__(self):
        self.enabled = config.USE_LLM
        self.client = None
        self.cache: dict[str, str] = {}
        self.last_error: str | None = None
        self.calls = 0
        if self.enabled:
            from openai import OpenAI
            self.client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY,
                                 timeout=config.LLM_TIMEOUT_S, max_retries=1)

    def _extra(self) -> dict:
        # gpt-oss models "think" before answering; keep that short to save tokens and time
        return {"reasoning_effort": "low"} if "gpt-oss" in config.LLM_MODEL else {}

    def complete(self, system: str, user: str, json_mode: bool = False,
                 max_tokens: int = 700, temperature: float = 0.1) -> str | None:
        if not self.enabled:
            return None
        key = hashlib.sha256(f"{json_mode}|{system}|{user}".encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        kwargs = dict(model=config.LLM_MODEL, temperature=temperature, max_tokens=max_tokens,
                      messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                      extra_body=self._extra() or None)
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        t0 = time.perf_counter()
        try:
            resp = self.client.chat.completions.create(**kwargs)
            text = (resp.choices[0].message.content or "").strip()
            self.calls += 1
            log.info("LLM call %.0f ms", (time.perf_counter() - t0) * 1000)
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"{exc.__class__.__name__}: {exc}"
            log.warning("LLM unavailable: %s", self.last_error)
            return None
        if text:
            self.cache[key] = text
        return text or None

    def complete_json(self, system: str, user: str, max_tokens: int = 700) -> dict | None:
        text = self.complete(system, user, json_mode=True, max_tokens=max_tokens, temperature=0)
        if not text:
            return None
        m = re.search(r"\{.*\}", text, re.S)          # tolerate text around the JSON
        try:
            return json.loads(m.group(0) if m else text)
        except (json.JSONDecodeError, AttributeError):
            self.last_error = "LLM returned invalid JSON"
            return None

    def status(self) -> dict:
        return {"enabled": self.enabled, "model": config.LLM_MODEL if self.enabled else None,
                "base_url": config.LLM_BASE_URL if self.enabled else None,
                "calls": self.calls, "last_error": self.last_error}


_llm: LLM | None = None


def get_llm() -> LLM:
    global _llm
    if _llm is None:
        _llm = LLM()
    return _llm
