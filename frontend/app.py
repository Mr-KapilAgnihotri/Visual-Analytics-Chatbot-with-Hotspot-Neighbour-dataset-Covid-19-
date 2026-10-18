"""
Chat interface (Streamlit).

Run:  streamlit run frontend/app.py

It sends each question to the FastAPI backend (POST /ask). If the backend is
not running, it answers IN-PROCESS by importing the same orchestrator, so the
demo keeps working either way.
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pandas as pd
import plotly.io as pio
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend import config  # noqa: E402

st.set_page_config(page_title="VA Chatbot – COVID-19 Germany", page_icon="📊", layout="wide")

# map outlines are served by Streamlit itself (frontend/static/topojson), no CDN needed
PLOTLY_CONFIG = {"topojsonURL": "app/static/topojson/", "displaylogo": False,
                 "modeBarButtonsToRemove": ["lasso2d", "select2d"]}
ENGINE_BADGE = {"SQL": ":blue-badge[SQL · DuckDB]", "GRAPH": ":violet-badge[Knowledge graph]",
                "HYBRID": ":orange-badge[Hybrid · graph + SQL]", "NONE": ":gray-badge[Assistant]"}
LANG = {"SQL": "sql", "Cypher": "cypher", "NetworkX (Python)": "python"}


# ---------------------------------------------------------------------------
# backend access
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the databases…")
def local_orchestrator():
    from backend.orchestrator import get_orchestrator
    return get_orchestrator()


def api_up() -> bool:
    try:
        return requests.get(f"{config.API_URL}/health", timeout=1.5).ok
    except requests.RequestException:
        return False


def ask(question: str) -> dict:
    """Send the question WITH this chat's session_id, so the backend remembers earlier turns."""
    sid = st.session_state["session_id"]
    if st.session_state.get("mode") == "api":
        try:
            r = requests.post(f"{config.API_URL}/ask", json={"question": question, "session_id": sid}, timeout=90)
            if r.ok:
                return r.json()
            return {"text": f"The API returned an error: {r.json().get('detail', r.text)}", "engine": "NONE"}
        except requests.RequestException:
            st.session_state["mode"] = "local"
    return local_orchestrator().ask(question, sid).to_dict()


def reset_memory() -> None:
    """Forget the conversation on the backend, then start a fresh session id."""
    sid = st.session_state.get("session_id")
    if sid:
        if st.session_state.get("mode") == "api":
            try:
                requests.post(f"{config.API_URL}/reset", json={"session_id": sid}, timeout=5)
            except requests.RequestException:
                pass
        else:
            local_orchestrator().memory.reset(sid)
    st.session_state["session_id"] = uuid.uuid4().hex


def status() -> dict:
    if st.session_state.get("mode") == "api":
        try:
            return requests.get(f"{config.API_URL}/health", timeout=2).json()
        except requests.RequestException:
            pass
    from backend.orchestrator import system_status
    local_orchestrator()
    return system_status()


if "mode" not in st.session_state:
    st.session_state["mode"] = "api" if api_up() else "local"
if "history" not in st.session_state:
    st.session_state["history"] = []
if "session_id" not in st.session_state:
    st.session_state["session_id"] = uuid.uuid4().hex     # one conversation = one memory


# ---------------------------------------------------------------------------
# rendering one answer
# ---------------------------------------------------------------------------
def render_answer(a: dict, key: str) -> None:
    if a.get("followup"):
        st.caption("↪ " + a["followup"])                     # how a follow-up was understood
    st.markdown(ENGINE_BADGE.get(a.get("engine", "NONE"), ""))
    st.markdown(a.get("text", ""))
    if a.get("kpis"):
        cols = st.columns(len(a["kpis"]))
        for c, k in zip(cols, a["kpis"]):
            c.metric(k["label"], k["value"])
    if a.get("chart"):
        fig = pio.from_json(a["chart"])
        st.plotly_chart(fig, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG, key=f"chart-{key}")
    if a.get("table"):
        with st.expander("Data behind the chart"):
            st.dataframe(pd.DataFrame(a["table"]), use_container_width=True, hide_index=True)
    if a.get("queries") or a.get("plan"):
        with st.expander("How this was answered"):
            plan, t = a.get("plan", {}), a.get("timings", {})
            st.markdown(
                f"**Understood as** `{a.get('intent')}` (by {'the LLM' if a.get('nlu_source') == 'llm' else 'rules'})"
                f" · **Engine** {a.get('engine')} · **Graph backend** {a.get('graph_backend')}"
                f" · **Text written by** {'the LLM' if a.get('narrative_source') == 'llm' else 'template'}")
            if t:
                st.caption(" · ".join(f"{k.replace('_ms', '')}: {v} ms" for k, v in t.items()))
            for q in a.get("queries", []):
                st.markdown(f"**{q['engine']}** – {q['purpose']} · {q['ms']} ms · {q['rows']} rows")
                st.code(q["text"], language=LANG.get(q["language"], "text"))
            with st.popover("Query plan (JSON)"):
                st.json(plan)


# ---------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📊 VA Chatbot")
    st.caption("COVID-19 in Germany's 411 districts, 2020 · Knowledge graph + relational database")
    s = status()
    g, llm = s.get("graph", {}), s.get("llm", {})
    st.markdown(f"**Mode:** {'FastAPI backend' if st.session_state['mode'] == 'api' else 'in-process'}")
    st.markdown(f"**Graph:** {'Neo4j ✅' if g.get('backend') == 'neo4j' else 'NetworkX (Neo4j not connected)'}")
    if g.get("fallback_reason"):
        st.caption(g["fallback_reason"][:200])
    st.markdown(f"**LLM:** {llm.get('model') + ' ✅' if llm.get('enabled') else 'off – rule-based parser'}")
    if llm.get("last_error"):
        st.caption(f"Last LLM error: {llm['last_error'][:160]}")
    st.divider()
    from backend.main import EXAMPLES
    for group, qs in EXAMPLES.items():
        st.markdown(f"**{group}**")
        for q in qs:
            if st.button(q, key=f"ex-{q}", use_container_width=True):
                st.session_state["pending"] = q
    st.divider()
    st.caption("Follow-ups work: try \"And on 1 December?\", \"Show their trend\", \"What about Hamburg?\".")
    if st.button("New topic (clear conversation and memory)", use_container_width=True):
        st.session_state["history"] = []
        reset_memory()
        st.rerun()

# ---------------------------------------------------------------------------
# main chat
# ---------------------------------------------------------------------------
st.title("Ask about COVID-19 in Germany, 2020")
if not st.session_state["history"]:
    st.info("Ask in plain English, e.g. *\"Which districts border Köln?\"* or "
            "*\"Top 10 districts by incidence on 15 November\"*. Examples are in the sidebar.")

for i, turn in enumerate(st.session_state["history"]):
    with st.chat_message("user"):
        st.markdown(turn["q"])
    with st.chat_message("assistant"):
        render_answer(turn["a"], str(i))

question = st.chat_input("Ask a question about the data…")
question = question or st.session_state.pop("pending", None)
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking: parsing → querying → charting → explaining…"):
            answer = ask(question)
        render_answer(answer, str(len(st.session_state["history"])))
    st.session_state["history"].append({"q": question, "a": answer})
