"""
REST API (FastAPI).

Endpoints
    POST /ask            {"question": "...", "session_id": "..."?}  -> full answer (text, chart JSON,
                         table, queries, timings, session_id, followup). Send the returned
                         session_id with the next question to enable follow-ups.
    POST /reset          {"session_id": "..."}  -> forget that conversation
    GET  /health         which graph backend and LLM are active
    GET  /examples       example questions grouped by engine
    GET  /topojson/...   map outlines served locally, so maps need no external CDN
    GET  /docs           interactive API documentation (generated automatically by FastAPI)

Run:  uvicorn backend.main:app --reload
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend import config
from backend.orchestrator import get_orchestrator, system_status

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

EXAMPLES = {
    "SQL (DuckDB)": [
        "Top 10 districts by 7-day incidence on 15 November",
        "Share of new cases by federal state",
        "How did new cases develop in Bavaria and Saxony weekly?",
        "Does mobility correlate with incidence during the second wave?",
        "Map of incidence on 20 December",
        "Number of hotspots over time",
        "Tell me about Leipzig",
    ],
    "Graph (Neo4j)": [
        "Which districts border Köln?",
        "Shortest path from Flensburg to Passau",
        "Which districts have the most neighbours?",
        "Which states border Hessen?",
    ],
    "Hybrid (both)": [
        "Districts within 2 borders of Berlin Mitte with incidence above 100",
        "Compare Köln with its neighbours' incidence",
        "Clusters of connected hotspots on 1 November",
        "Hotspots surrounded by hotspots on 1 November",
        "Isolated hotspots on 15 October",
    ],
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_orchestrator()          # warm up: open DuckDB, connect to Neo4j, build the entity index
    yield


app = FastAPI(title="Visual Analytics Chatbot – COVID-19 Germany 2020",
              description="Hybrid knowledge-graph (Neo4j) + relational (DuckDB) question answering.",
              version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/topojson", StaticFiles(directory=config.ROOT / "frontend" / "static" / "topojson"), name="topojson")


class Question(BaseModel):
    question: str = Field(..., min_length=2, max_length=500, examples=["Which districts border Köln?"])
    session_id: str | None = Field(None, max_length=64,
                                   description="Conversation id; omit to start a new conversation")


class Session(BaseModel):
    session_id: str = Field(..., max_length=64)


@app.post("/ask")
def ask(q: Question) -> dict:
    orch = get_orchestrator()
    session_id = q.session_id or orch.memory.new_session_id()
    try:
        return orch.ask(q.question.strip(), session_id).to_dict()
    except Exception as exc:  # noqa: BLE001
        logging.exception("Failed to answer %r", q.question)
        raise HTTPException(status_code=500, detail=f"{exc.__class__.__name__}: {exc}") from exc


@app.post("/reset")
def reset(s: Session) -> dict:
    """Forget a conversation, so the next question starts a new topic."""
    return {"session_id": s.session_id, "reset": get_orchestrator().memory.reset(s.session_id)}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", **system_status()}


@app.get("/examples")
def examples() -> dict:
    return EXAMPLES
