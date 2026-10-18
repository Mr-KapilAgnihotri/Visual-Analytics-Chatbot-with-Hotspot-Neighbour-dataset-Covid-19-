"""
Tests for conversation memory (follow-up questions).   Run:  python -m pytest -q

They use the rule-based path only (no LLM), so they run offline and give the
same result every time.
"""
import logging
import uuid

import pytest

logging.disable(logging.WARNING)

from backend.memory import ConversationStore, Turn  # noqa: E402
from backend.orchestrator import get_orchestrator  # noqa: E402


@pytest.fixture(scope="module")
def orch():
    return get_orchestrator()


def sid() -> str:
    return uuid.uuid4().hex


def places(a) -> list:
    return [d["name"] for d in a.plan["districts"]] + a.plan["states"]


# ---------------------------------------------------------------- the six-turn dialogue
def test_six_turn_dialogue(orch):
    s = sid()
    a1 = orch.ask("Top 10 districts by incidence on 15 November", s)
    assert (a1.intent, a1.plan["metric"], a1.plan["date"], a1.plan["n"]) == ("top_n", "incidence_7d", "2020-11-15", 10)
    assert a1.followup is None

    a2 = orch.ask("And on 1 December?", s)                         # ellipsis: only the date changes
    assert (a2.intent, a2.plan["metric"], a2.plan["date"], a2.plan["n"]) == ("top_n", "incidence_7d", "2020-12-01", 10)
    assert a2.followup and "1 Dec 2020" in a2.followup
    top10 = [r["district_id"] for r in a2.table]

    a3 = orch.ask("Show their trend", s)                            # anaphora: "their" = the 10 districts
    assert a3.intent == "trend" and a3.plan["metric"] == "incidence_7d"
    ids3 = [d["id"] for d in a3.plan["districts"]]
    assert ids3 == top10[:8]                                       # one line each, at most 8

    a4 = orch.ask("Only the ones in Bavaria", s)                    # refinement: filter by state
    assert a4.intent == "trend"
    assert a4.plan["districts"] and all(
        orch.linker.district_state[d["id"]] == "Freistaat Bayern" for d in a4.plan["districts"])

    a5 = orch.ask("What about deaths?", s)                          # substitution of the measure
    assert a5.intent == "trend" and a5.plan["metric"] == "deaths"
    assert [d["id"] for d in a5.plan["districts"]] == [d["id"] for d in a4.plan["districts"]]

    a6 = orch.ask("Now as a map", s)                                # same data, new chart
    assert a6.intent == "snapshot_map" and a6.plan["metric"] == "deaths" and a6.chart


# ---------------------------------------------------------------- other follow-up kinds
def test_what_about_replaces_place(orch):
    s = sid()
    orch.ask("Which districts border Köln?", s)
    a = orch.ask("What about Hamburg?", s)
    assert a.intent == "neighbours" and places(a) == ["Hamburg"]


def test_its_neighbours(orch):
    s = sid()
    orch.ask("Tell me about Leipzig", s)
    a = orch.ask("its neighbours", s)
    assert a.intent == "neighbours" and places(a) == ["Leipzig, Stadt"]


def test_also_add_appends(orch):
    s = sid()
    orch.ask("Compare Bavaria and Saxony monthly new cases", s)
    a = orch.ask("also add Hessen", s)
    assert a.intent == "trend" and set(a.plan["states"]) == {"Freistaat Bayern", "Sachsen", "Hessen"}
    assert a.plan["granularity"] == "month" and a.plan["metric"] == "new_cases"


def test_which_of_them_are_hotspots(orch):
    s = sid()
    a1 = orch.ask("Top 10 districts by new cases in November", s)
    a2 = orch.ask("Which of them are hotspots?", s)
    assert a2.intent == "subset" and a2.plan["metric"] == "incidence_7d"
    assert sorted(d["id"] for d in a2.plan["districts"]) == sorted(r["district_id"] for r in a1.table)
    assert "hotspots" in a2.facts


def test_answer_to_clarification(orch):
    s = sid()
    assert orch.ask("neighbours of", s).engine == "NONE"            # the bot asks which district
    a = orch.ask("Köln", s)
    assert a.intent == "neighbours" and places(a) == ["Köln"]


def test_hops_and_path_followups(orch):
    s = sid()
    orch.ask("Districts within 2 borders of Jena", s)
    a = orch.ask("and within 3?", s)
    assert a.intent == "k_hop" and a.plan["hops"] == 3 and places(a) == ["Jena"]
    s2 = sid()
    orch.ask("Shortest path from Flensburg to Passau", s2)
    b = orch.ask("and to Köln?", s2)
    assert b.intent == "shortest_path" and places(b) == ["Flensburg, Stadt", "Köln"]


def test_complete_question_starts_fresh(orch):
    s = sid()
    orch.ask("Top 10 districts by incidence on 15 November", s)
    a = orch.ask("Which districts border Köln?", s)
    assert a.followup is None and a.intent == "neighbours"
    b = orch.ask("Map of incidence on 20 December", s)
    assert b.followup is None and b.plan["date"] == "2020-12-20" and not b.plan["districts"]


def test_no_session_means_no_memory(orch):
    orch.ask("Which districts border Köln?")
    a = orch.ask("What about Hamburg?")                             # no session id: stateless
    assert a.followup is None


# ---------------------------------------------------------------- the store itself
def test_sessions_are_isolated(orch):
    s1, s2 = sid(), sid()
    orch.ask("Which districts border Köln?", s1)
    a = orch.ask("What about Hamburg?", s2)                         # s2 has no history
    assert a.followup is None


def test_reset_clears_memory(orch):
    s = sid()
    orch.ask("Which districts border Köln?", s)
    assert orch.memory.reset(s) is True
    assert orch.ask("What about Hamburg?", s).followup is None


def test_store_limits_and_ttl():
    now = [1000.0]
    store = ConversationStore(max_turns=3, ttl_seconds=60, clock=lambda: now[0])
    for i in range(5):
        store.add("a", Turn(question=f"q{i}", plan={}))
    assert [t.question for t in store.history("a")] == ["q2", "q3", "q4"]   # only the last 3 kept
    now[0] += 61                                                             # an idle minute passes
    assert store.last("a") is None and len(store) == 0                       # forgotten (TTL)


# ---------------------------------------------------------------- the API
def test_api_session_and_reset():
    from fastapi.testclient import TestClient
    from backend.main import app
    with TestClient(app) as client:
        r1 = client.post("/ask", json={"question": "Which districts border Köln?"}).json()
        assert r1["session_id"]                                   # a new conversation gets an id
        r2 = client.post("/ask", json={"question": "What about Hamburg?", "session_id": r1["session_id"]}).json()
        assert r2["followup"] and r2["intent"] == "neighbours"
        r3 = client.post("/reset", json={"session_id": r1["session_id"]}).json()
        assert r3["reset"] is True
        r4 = client.post("/ask", json={"question": "What about Hamburg?", "session_id": r1["session_id"]}).json()
        assert r4["followup"] is None
