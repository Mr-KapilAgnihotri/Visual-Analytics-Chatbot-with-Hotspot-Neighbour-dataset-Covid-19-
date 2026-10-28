"""
CONVERSATION MEMORY: what the chatbot remembers between questions.

For each conversation (a "session", identified by a random session_id) we keep
the last few TURNS. A turn is small and structured, not the chat text:

    question      the user's words
    plan          the QueryPlan that was answered (intent, metric, places, dates ...)
    result_ids    the district IDs shown in that answer (so "their trend" can refer to them)
    result_states the federal states shown in that answer
    ts            when it happened

Why structured memory instead of the raw chat history?
  * it is tiny (a few hundred bytes per turn), so it costs no LLM tokens
  * it is exact: "their" means precisely the 10 IDs that were on screen
  * it is testable and explainable (dialogue-state tracking)

Design limits (a viva point: memory must be bounded):
  * at most MAX_TURNS turns per session (oldest dropped first)
  * a session idle for longer than TTL_SECONDS is forgotten (time-to-live)
  * memory lives in this process only; it is never written into covid.duckdb,
    which stays read-only. Restarting the API starts every conversation fresh.
"""
from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field

MAX_TURNS = 10
TTL_SECONDS = 60 * 60


@dataclass
class Turn:
    question: str
    plan: dict
    result_ids: list = field(default_factory=list)
    result_states: list = field(default_factory=list)
    ts: float = 0.0
    answer: dict = field(default_factory=dict)   # intent, chart title, facts, text: for "explain this chart"


class ConversationStore:
    """Thread-safe, in-memory store: session_id -> recent turns."""

    def __init__(self, max_turns: int = MAX_TURNS, ttl_seconds: float = TTL_SECONDS, clock=time.time):
        self.max_turns = max_turns
        self.ttl = ttl_seconds
        self.clock = clock                      # injectable, so tests can fake the passing of time
        self._sessions: dict[str, deque] = {}
        self._last_seen: dict[str, float] = {}
        self._lock = threading.Lock()

    @staticmethod
    def new_session_id() -> str:
        return uuid.uuid4().hex

    def _expire(self) -> None:
        """Forget sessions idle for longer than the TTL (called on every access)."""
        now = self.clock()
        for sid in [s for s, t in self._last_seen.items() if now - t > self.ttl]:
            self._sessions.pop(sid, None)
            self._last_seen.pop(sid, None)

    def add(self, session_id: str, turn: Turn) -> None:
        with self._lock:
            self._expire()
            turn.ts = self.clock()
            self._sessions.setdefault(session_id, deque(maxlen=self.max_turns)).append(turn)
            self._last_seen[session_id] = turn.ts

    def last(self, session_id: str | None) -> Turn | None:
        if not session_id:
            return None
        with self._lock:
            self._expire()
            turns = self._sessions.get(session_id)
            if not turns:
                return None
            self._last_seen[session_id] = self.clock()
            return turns[-1]

    def history(self, session_id: str) -> list[Turn]:
        with self._lock:
            self._expire()
            return list(self._sessions.get(session_id, []))

    def reset(self, session_id: str) -> bool:
        with self._lock:
            existed = session_id in self._sessions
            self._sessions.pop(session_id, None)
            self._last_seen.pop(session_id, None)
            return existed

    def __len__(self) -> int:
        with self._lock:
            self._expire()
            return len(self._sessions)
