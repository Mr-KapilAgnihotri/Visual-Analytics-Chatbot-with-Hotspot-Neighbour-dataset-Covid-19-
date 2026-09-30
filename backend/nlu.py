"""
NATURAL-LANGUAGE UNDERSTANDING: question text -> a structured QueryPlan.

Two layers, deliberately:

  1. Deterministic layer (always runs, ~1 ms)
       * EntityLinker finds districts, states, dates and numbers
       * a keyword rule-set guesses the intent and metric
  2. LLM layer (Groq; used when a key is configured)
       * receives the question PLUS what layer 1 found, and returns a small JSON
         object: intent, metric(s), level, n, order, hops ...
       * its answer is VALIDATED against the catalogue; anything invalid is
         ignored and the rule-based value is kept

The LLM never writes SQL or Cypher and never sees the data. It only fills in a
form, which is cheap in tokens, fast, and cannot produce a dangerous query.
If the LLM is unavailable (no key, offline, rate-limited) the rules alone answer.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

from backend import config
from backend.catalog import INTENTS, METRICS
from backend.entities import EntityLinker, Extraction, fold
from backend.llm import get_llm


@dataclass
class QueryPlan:
    intent: str = "top_n"
    metric: str = "incidence_7d"
    metric2: str | None = None
    level: str = "district"                 # district | state
    districts: list = field(default_factory=list)   # [{"id":..,"name":..}]
    states: list = field(default_factory=list)      # [state names]
    date: str | None = None
    start: str | None = None
    end: str | None = None
    granularity: str = "day"
    n: int = 10
    order: str = "desc"
    hops: int = 2
    peak: bool = False
    threshold: float | None = None          # e.g. "incidence above 100"
    nlu_source: str = "rules"
    notes: list = field(default_factory=list)
    clarification: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Rule-based intent detection (ordered: first match wins)
# ---------------------------------------------------------------------------
NB = r"(neighbou?rs?|neighbou?ring|border(s|ing)?|adjacent|next to|surrounding)"
RULES = [
    ("help",               r"\b(help|what can you|what questions|examples?)\b"),
    ("shortest_path",      r"\b(shortest|path|route|how many borders|border crossings|hops? (from|between))\b"),
    ("surrounded_hotspots", r"\b(surrounded|encircled|all (of )?(its|their) neighbou?rs)\b"),
    ("isolated_hotspots",  r"\b(isolated|no hotspot neighbou?rs?|without hotspot neighbou?rs?|lone)\b"),
    ("hotspot_clusters",   r"\b(clusters?|connected hotspots?|groups? of hotspots?|adjacent hotspots?|hotspot regions?)\b"),
    ("most_connected",     r"\b(most (neighbou?rs|connected|borders|bordering)|degree|centrality|most central)\b"),
    ("k_hop",              r"\b(within \d+|\d+[- ]?(hops?|borders?|steps?)|neighbou?rs of neighbou?rs|second[- ]degree|k[- ]hop)\b"),
    ("state_borders",      rf"\b(which|what) (federal )?states? {NB}\b|\bstates? {NB}\b"),
    ("neighbour_metrics",  rf"{NB}.*\b(incidence|cases|deaths|hotspot|compare|vs|versus|rate)\b|"
                           rf"\b(incidence|cases|deaths|compare|rate)\b.*{NB}"),
    ("neighbours",         rf"\b{NB}\b"),
    ("correlation",        r"\b(correlat\w*|relationship|related|relation|impact|effect|affect\w*|associated)\b"),
    ("hotspot_trend",      r"\b(how many|number of|count of) hotspots?\b|\bhotspots? (over time|trend|per day|by day)\b"),
    ("share",              r"\b(share|proportion|percentage|percent|distribution|breakdown|pie|split)\b"),
    ("snapshot_map",       r"\b(map|where|geograph\w*|spatial\w*|across germany)\b"),
    ("compare",            r"\b(compare|comparison|vs\.?|versus|against|difference between)\b"),
    ("trend",              r"\b(trend|over time|evolution|evolve\w*|develop\w*|progress\w*|timeline|curve|daily|weekly|monthly|per (day|week|month)|time series|how did)\b"),
    ("top_n",              r"\b(top|highest|most|lowest|least|fewest|rank\w*|worst|best|biggest|largest|smallest|which (district|state)s?)\b"),
    ("profile",            r"\b(tell me about|profile|summary|overview|facts about|situation in|about)\b"),
]
RULE = dict(RULES)
LEVEL_STATE = r"\b(states|federal states|bundeslander|by state|per state|each state|state level)\b"


def _metric_mentions(text: str) -> list[str]:
    """Metrics mentioned in the text, in order of appearance (longest synonym first)."""
    found = []
    for key, m in METRICS.items():
        for syn in sorted(m["synonyms"], key=len, reverse=True):
            for hit in re.finditer(rf"\b{re.escape(syn)}\b", text):
                found.append((hit.start(), len(syn), key))
    found.sort(key=lambda f: (f[0], -f[1]))
    out, covered = [], -1
    for pos, ln, key in found:
        if pos > covered and key not in out:
            out.append(key)
            covered = pos + ln - 1
    return out


DEFAULT_METRIC = {
    "top_n": "incidence_7d", "trend": "new_cases", "share": "new_cases", "compare": "incidence_7d",
    "snapshot_map": "incidence_7d", "neighbour_metrics": "incidence_7d", "k_hop": "incidence_7d",
    "correlation": "mobility", "profile": "incidence_7d",
}


def rule_plan(question: str, ex: Extraction) -> QueryPlan:
    q = question.lower()
    fq = fold(question)
    p = QueryPlan()
    for intent, pattern in RULES:
        if re.search(pattern, q) or re.search(pattern, fq):
            p.intent = intent
            break
    else:
        has_place = bool(ex.districts or ex.states)
        p.intent = ("trend" if _metric_mentions(fq) else "profile") if has_place else "top_n"

    # a lone place name with nothing else -> profile
    if p.intent == "top_n" and not re.search(RULE["top_n"], q) and len(ex.districts) + len(ex.states) == 1:
        p.intent = "profile"
    # "compare" with only one place and neighbour words handled above; with 0-1 places -> top_n
    if p.intent == "compare" and len(ex.districts) + len(ex.states) < 2:
        p.intent = "top_n" if not (ex.districts or ex.states) else "trend"
    if p.intent == "hotspot_trend" and ex.date and not (ex.start or ex.end):
        p.intent = "snapshot_map"

    metrics = _metric_mentions(fq)
    if p.intent == "correlation":
        p.metric = metrics[0] if metrics else "mobility"
        p.metric2 = metrics[1] if len(metrics) > 1 else ("incidence_7d" if p.metric != "incidence_7d" else "mobility")
    else:
        p.metric = metrics[0] if metrics else DEFAULT_METRIC.get(p.intent, "incidence_7d")
        if p.metric == "hotspot" and p.intent in ("trend",):
            p.intent = "hotspot_trend"

    p.level = "state" if re.search(LEVEL_STATE, fq) and not ex.districts else "district"
    p.order = "asc" if re.search(r"\b(lowest|least|fewest|smallest|best|minimum|bottom)\b", q) else "desc"
    p.peak = bool(re.search(r"\b(peak|maximum|max|highest ever)\b", q))
    if re.search(r"\b(weekly|per week|by week|each week)\b", q):
        p.granularity = "week"
    elif re.search(r"\b(monthly|per month|by month|each month)\b", q):
        p.granularity = "month"

    if p.intent == "compare" and (p.granularity != "day" or re.search(RULE["trend"], q)):
        p.intent = "trend"                   # "compare X and Y weekly / over time" -> multi-line trend
    t = re.search(r"\b(?:above|over|greater than|more than|exceed\w*|at least|>=?)\s*(\d+(?:\.\d+)?)", q)
    if t:
        p.threshold = float(t[1])

    m = re.search(r"\b(?:top|bottom|first|lowest|highest)\s+(\d{1,2})\b|\b(\d{1,2})\s+(?:districts|states|places)\b", q)
    if m:
        p.n = int(m[1] or m[2])
    h = re.search(r"\bwithin\s+(\d)\b|\b(\d)[- ]?(?:hops?|borders?|steps?)\b", q)
    if h:
        p.hops = int(h[1] or h[2])
    elif re.search(r"neighbou?rs of neighbou?rs|second[- ]degree", q):
        p.hops = 2

    p.districts = [{"id": int(d.id), "name": d.name} for d in ex.districts]
    p.states = [s.name for s in ex.states]
    p.date, p.start, p.end = ex.date, ex.start, ex.end
    p.notes = list(ex.notes)
    return p


# ---------------------------------------------------------------------------
# LLM layer
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You convert questions about a COVID-19 dataset of 411 German districts
(daily data, {first} to {last}) into a JSON query plan. Reply with JSON only.

intents:
{intents}

metrics: {metrics}

JSON keys (omit any you are unsure of):
 intent (one of the intents), metric, metric2 (only for correlation),
 level ("district" or "state"), places (list of place names from the question),
 date (YYYY-MM-DD, a single day), start, end (YYYY-MM-DD, a period),
 granularity ("day","week","month"), n (int, how many results), order ("desc","asc"),
 hops (int 1-6), peak (true if asking for the maximum over a period),
 threshold (number, e.g. 100 for "incidence above 100")
Hotspot means 7-day incidence >= 75 per 100k."""


def _llm_system() -> str:
    intents = "\n".join(f" {k}: {v[1]}" for k, v in INTENTS.items())
    metrics = ", ".join(f"{k} ({v['label']})" for k, v in METRICS.items())
    return SYSTEM_PROMPT.format(first=config.FIRST_DATE, last=config.LAST_DATE,
                                intents=intents, metrics=metrics)


def _valid_date(s) -> str | None:
    if isinstance(s, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return min(max(s, config.FIRST_DATE), config.LAST_DATE)
    return None


def merge_llm(plan: QueryPlan, js: dict, linker: EntityLinker, question: str) -> QueryPlan:
    """Overlay validated LLM fields on the rule-based plan."""
    p = QueryPlan(**plan.to_dict())
    if js.get("intent") in INTENTS:
        p.intent = js["intent"]
    if js.get("metric") in METRICS:
        p.metric = js["metric"]
    if js.get("metric2") in METRICS:
        p.metric2 = js["metric2"]
    if js.get("level") in ("district", "state"):
        p.level = js["level"]
    if js.get("granularity") in ("day", "week", "month"):
        p.granularity = js["granularity"]
    if js.get("order") in ("asc", "desc"):
        p.order = js["order"]
    for key, lo, hi in (("n", 1, 50), ("hops", 1, 6)):
        if isinstance(js.get(key), int) and lo <= js[key] <= hi:
            setattr(p, key, js[key])
    if isinstance(js.get("threshold"), (int, float)) and not plan.threshold:
        p.threshold = float(js["threshold"])
    if isinstance(js.get("peak"), bool):
        p.peak = js["peak"]
    # dates: trust the deterministic parser when it found something
    if not (plan.date or plan.start):
        d, s, e = _valid_date(js.get("date")), _valid_date(js.get("start")), _valid_date(js.get("end"))
        if s and e and s <= e:
            p.start, p.end = s, e
        elif d:
            p.date = d
    # places: re-link LLM names only if the linker found nothing itself
    places = js.get("places")
    if isinstance(places, list) and not (plan.districts or plan.states):
        ex = linker.extract(" , ".join(str(x) for x in places))
        p.districts = [{"id": int(d.id), "name": d.name} for d in ex.districts]
        p.states = [s.name for s in ex.states]
    if p.intent == "correlation" and not p.metric2:
        p.metric2 = "incidence_7d" if p.metric != "incidence_7d" else "mobility"
    p.nlu_source = "llm"
    return p


def understand(question: str, linker: EntityLinker) -> tuple[QueryPlan, Extraction]:
    ex = linker.extract(question)
    plan = rule_plan(question, ex)
    llm = get_llm()
    if llm.enabled:
        found = {"districts": [d.name for d in ex.districts], "states": [s.name for s in ex.states],
                 "date": ex.date, "start": ex.start, "end": ex.end}
        user = f"Question: {question}\nAlready detected: {json.dumps(found, ensure_ascii=False)}"
        js = llm.complete_json(_llm_system(), user, max_tokens=600)
        if js:
            plan = merge_llm(plan, js, linker, question)
    return finalize(plan), ex


def finalize(p: QueryPlan) -> QueryPlan:
    """Fill defaults and ask for clarification when a required place is missing."""
    one_place = {"neighbours", "k_hop", "neighbour_metrics"}
    if p.intent in one_place and not p.districts:
        if p.states and p.intent == "neighbours":
            p.intent = "state_borders"
        else:
            p.clarification = "Which district do you mean? For example: \"neighbours of Köln\"."
    if p.intent == "shortest_path" and len(p.districts) < 2:
        p.clarification = "Please name two districts, e.g. \"shortest path from Flensburg to Passau\"."
    if p.intent == "state_borders" and not p.states:
        if p.districts:
            p.intent = "neighbours"
        else:
            p.clarification = "Which federal state? For example: \"which states border Hessen\"."
    if p.intent == "profile" and not (p.districts or p.states):
        p.intent = "top_n"
    if p.intent == "compare" and len(p.districts) + len(p.states) < 2:
        p.intent = "trend" if (p.districts or p.states) else "top_n"
    if p.metric == "hotspot" and p.intent in ("share", "snapshot_map", "compare"):
        p.metric = "incidence_7d"
    p.n = max(1, min(p.n, 50))
    p.hops = max(1, min(p.hops, 6))
    return p
