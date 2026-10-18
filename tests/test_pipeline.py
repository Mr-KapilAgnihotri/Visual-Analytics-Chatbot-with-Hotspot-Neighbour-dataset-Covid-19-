"""
Automated tests.   Run:  python -m pytest -q

They check the data (borders are exact and symmetric), the language layer
(entities, dates, intents), every question type end to end, safety against
SQL injection, and the LLM merge logic (with a fake LLM, so no key is needed).
"""
import json
import logging

import pytest

logging.disable(logging.WARNING)

from backend import config  # noqa: E402
from backend.catalog import INTENTS  # noqa: E402
from backend.entities import EntityLinker  # noqa: E402
from backend.graph_engine import NetworkXGraph  # noqa: E402
from backend.nlu import QueryPlan, merge_llm, rule_plan  # noqa: E402
from backend.orchestrator import get_orchestrator  # noqa: E402
from backend.sql_engine import get_sql  # noqa: E402


@pytest.fixture(scope="session")
def orch():
    return get_orchestrator()


@pytest.fixture(scope="session")
def linker():
    return EntityLinker(get_sql().districts())


# ---------------------------------------------------------------- data
def test_row_and_district_counts():
    sql = get_sql()
    assert sql.run("SELECT COUNT(DISTINCT district_id) n FROM dim_district").df["n"][0] == 411
    assert sql.run("SELECT COUNT(*) n FROM dim_state").df["n"][0] == 16
    assert sql.run("SELECT COUNT(*) n FROM fact_daily").df["n"][0] == 149_592


def test_borders_symmetric_and_verified():
    adj = json.loads(config.ADJ_JSON.read_text())
    a = {int(k): set(v) for k, v in adj["adjacency"].items()}
    assert all(i in a[j] for i, ns in a.items() for j in ns), "border graph must be symmetric"
    assert all(e == 0 for e in adj["residual_error"].values()), "every district solved exactly"


def test_hotspot_rule_matches_threshold():
    df = get_sql().run("SELECT hotspot, incidence_7d FROM fact_daily").df
    assert ((df["incidence_7d"] >= config.HOTSPOT_THRESHOLD) == df["hotspot"]).all()


def test_munich_name_fix():
    df = get_sql().run("SELECT district FROM dim_district WHERE district_id = 9162").df
    assert df["district"][0] == "München, Landeshauptstadt"


# ---------------------------------------------------------------- language layer
@pytest.mark.parametrize("q,expected", [
    ("neighbours of Munich", "München, Landeshauptstadt"),
    ("neighbours of Landkreis Augsburg", "Augsburg, Kreis"),
    ("incidence in Dusseldorff", "Düsseldorf"),
    ("Brandenburg an der Havel", "Brandenburg an der Havel"),
    ("cases in Neukölln", "Berlin Neukölln"),
])
def test_entity_linking(linker, q, expected):
    assert linker.extract(q).districts[0].name == expected


@pytest.mark.parametrize("q,date,start,end", [
    ("top districts on 15 November", "2020-11-15", None, None),
    ("cases in November", None, "2020-11-01", "2020-11-30"),
    ("between March and May", None, "2020-03-01", "2020-05-31"),
    ("on 2020-12-01", "2020-12-01", None, None),
    ("during the second wave", None, "2020-10-01", "2020-12-31"),
])
def test_dates(linker, q, date, start, end):
    ex = linker.extract(q)
    assert (ex.date, ex.start, ex.end) == (date, start, end)


@pytest.mark.parametrize("q,intent", [
    ("Top 10 districts by incidence on 15 November", "top_n"),
    ("Share of cases by federal state", "share"),
    ("How did new cases develop in Bavaria?", "trend"),
    ("Does mobility correlate with incidence?", "correlation"),
    ("Which districts border Köln?", "neighbours"),
    ("Shortest path from Flensburg to Passau", "shortest_path"),
    ("Districts within 2 borders of Jena", "k_hop"),
    ("Compare Köln with its neighbours' incidence", "neighbour_metrics"),
    ("Clusters of hotspots on 1 December", "hotspot_clusters"),
    ("Hotspots surrounded by hotspots on 1 December", "surrounded_hotspots"),
    ("Which states border Hessen?", "state_borders"),
    ("Tell me about Leipzig", "profile"),
])
def test_rule_intents(linker, q, intent):
    assert rule_plan(q, linker.extract(q)).intent == intent


def test_llm_merge_validates(linker):
    """A fake LLM answer: valid fields fill the gaps, invalid ones are ignored."""
    q = "munich stuff"                                     # the rules recognise no question type here
    base = rule_plan(q, linker.extract(q))
    fake = {"intent": "k_hop", "metric": "new_cases", "hops": 3, "order": "sideways", "n": 999,
            "places": ["Munich"], "date": "2021-05-01"}
    p = merge_llm(base, fake, linker, "q")
    assert (p.intent, p.metric, p.hops, p.nlu_source) == ("k_hop", "new_cases", 3, "llm")
    assert p.order == "desc" and p.n == base.n            # invalid values ignored
    assert p.date == config.LAST_DATE                     # out-of-range date clamped


def test_llm_never_overrules_the_rules(linker):
    """Regression: with Groq on, the LLM turned "When did Germany have the most new cases?"
    (a timeline) into a ranking. Slots the rules recognised must win."""
    q = "When did Germany have the most new cases?"
    base = rule_plan(q, linker.extract(q))
    p = merge_llm(base, {"intent": "top_n", "metric": "deaths"}, linker, q)
    assert (p.intent, p.metric) == ("trend", "new_cases")


def test_llm_keeps_type_in_followup(orch, monkeypatch):
    """Regression: in a conversation, the LLM turned "also add Hessen" (a monthly trend)
    into a comparison. A question type only the LLM saw must not replace the remembered one."""
    import uuid
    from backend import nlu

    class FakeLLM:
        enabled = True

        def complete_json(self, *a, **k):
            return {"intent": "compare", "places": ["Hessen"]}

    s = uuid.uuid4().hex
    orch.ask("Compare Bavaria and Saxony monthly new cases", s)
    monkeypatch.setattr(nlu, "get_llm", lambda: FakeLLM())
    a = orch.ask("also add Hessen", s)
    assert a.intent == "trend" and set(a.plan["states"]) == {"Freistaat Bayern", "Sachsen", "Hessen"}


# ---------------------------------------------------------------- end to end
QUESTIONS = {
    "top_n": "Top 10 districts by 7-day incidence on 15 November",
    "trend": "How did new cases develop in Bavaria and Saxony weekly?",
    "share": "Share of new cases by federal state",
    "compare": "Compare Munich and Hamburg",
    "correlation": "Does mobility correlate with incidence during the second wave?",
    "hotspot_trend": "Number of hotspots over time",
    "snapshot_map": "Map of incidence on 20 December",
    "profile": "Tell me about Leipzig",
    "neighbours": "Which districts border Köln?",
    "shortest_path": "Shortest path from Flensburg to Passau",
    "most_connected": "Which districts have the most neighbours?",
    "state_borders": "Which states border Hessen?",
    "k_hop": "Districts within 2 borders of Berlin Mitte with incidence above 100",
    "neighbour_metrics": "Compare Köln with its neighbours' incidence",
    "hotspot_clusters": "Clusters of connected hotspots on 1 November",
    "surrounded_hotspots": "Hotspots surrounded by hotspots on 1 November",
    "isolated_hotspots": "Isolated hotspots on 15 October",
}


@pytest.mark.parametrize("intent", list(QUESTIONS))
def test_every_intent_end_to_end(orch, intent):
    a = orch.ask(QUESTIONS[intent])
    assert a.intent == intent
    assert a.text and a.chart, "every analytical answer has text and a chart"
    expected = INTENTS[intent][0]
    if intent != "profile":                    # profile uses both engines by design
        assert a.engine == expected
    json.dumps(a.to_dict())                    # serialisable for the API


def test_sql_injection_is_harmless(orch):
    a = orch.ask("neighbours of Köln'; DROP TABLE fact_daily; --")
    assert a.intent == "neighbours"
    assert get_sql().run("SELECT COUNT(*) n FROM fact_daily").df["n"][0] == 149_592


def test_graph_counts():
    g = NetworkXGraph()
    assert len(g.neighbours(5315).rows) == 6                  # Köln
    assert len(g.shortest_path(1001, 9262).rows) - 1 == 23    # Flensburg -> Passau (city)


def test_narration_grounding():
    from backend.narrator import ungrounded_numbers
    facts = {"value": 184.5, "average": 162.0, "period": "on 15 Nov 2020", "total": 1734471}
    assert ungrounded_numbers("It had 184.5 vs 162 on 15 November 2020, about 1.7 million in total.", facts) == []
    assert ungrounded_numbers("It had 190, 14% more than average.", facts) == ["190", "14"]


# ---------------------------------------------------------------- regressions found by the question battery
@pytest.mark.parametrize("q,intent", [
    ("How many districts were hotspots each day during the second wave?", "hotspot_trend"),
    ("Which state shares the most borders with Niedersachsen?", "state_borders"),
    ("Which hotspots on 15 October had no neighbouring hotspot?", "isolated_hotspots"),
    ("When did Germany have the most new cases?", "trend"),
    ("When did Köln first become a hotspot?", "profile"),
])
def test_battery_intents(orch, q, intent):
    assert orch.ask(q).intent == intent


def test_period_map_uses_period_data(orch):
    """Regression: the April map once showed 31 December data under an April label."""
    a = orch.ask("Where was COVID worst in April?")
    assert "30 Apr 2020" in a.facts["period"] and a.facts["max"] < 150


def test_flow_without_date_means_whole_year(orch):
    a = orch.ask("Compare deaths in Bavaria and Baden-Württemberg")
    assert a.plan["start"] == config.FIRST_DATE and a.facts["values"][0]["value"] > 5000


def test_holiday_date(linker):
    assert linker.extract("lowest incidence on Christmas day").date == "2020-12-25"
