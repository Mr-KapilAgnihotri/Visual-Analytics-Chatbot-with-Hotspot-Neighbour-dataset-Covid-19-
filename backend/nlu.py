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
    # which slots the user actually STATED in this question (vs. defaults); conversation
    # memory uses this: stated slots override the previous turn, unstated ones are inherited
    explicit: list = field(default_factory=list)
    followup: str | None = None             # human-readable note when this was a follow-up

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
    ("isolated_hotspots",  r"\b(isolated|lone)\b|\b(no|without( any)?|not a single)\s+(neighbou?ring|adjacent|bordering)?\s*hotspots?\s*neighbou?rs?\b|"
                           r"\b(no|without( any)?)\s+(neighbou?ring|adjacent|bordering)\s+hotspots?\b"),
    ("hotspot_clusters",   r"\b(clusters?|connected hotspots?|groups? of hotspots?|adjacent hotspots?|hotspot regions?)\b"),
    ("most_connected",     r"\b(most (neighbou?rs|connected|borders|bordering)|degree|centrality|most central)\b"),
    ("k_hop",              r"\b(within \d+|\d+[- ]?(hops?|borders?|steps?)|neighbou?rs of neighbou?rs|second[- ]degree|k[- ]hop)\b"),
    ("state_borders",      rf"\b(which|what) (federal )?states? {NB}\b|\bstates? {NB}\b|"
                           rf"\b(which|what) (federal )?states? (shares?|has|have) (the )?(most|longest|any)?\s*borders?\b"),
    ("neighbour_metrics",  rf"{NB}.*\b(incidence|cases|deaths|hotspot|compare|vs|versus|rate)\b|"
                           rf"\b(incidence|cases|deaths|compare|rate)\b.*{NB}"),
    ("neighbours",         rf"\b{NB}\b"),
    ("correlation",        r"\b(correlat\w*|relationship|related|relation|impact|effect|affect\w*|associated)\b"),
    ("hotspot_trend",      r"\b(how many|number of|count of)( districts)?( were| was| are)?( in)? hotspots?\b|"
                           r"\bhotspots? (over time|trend|per day|by day|each day|every day)\b|\bhotspots? each (day|week)\b"),
    ("share",              r"\b(share|proportion|percentage|percent|distribution|breakdown|pie|split)\b"),
    ("snapshot_map",       r"\b(map|where|geograph\w*|spatial\w*|across germany)\b"),
    ("compare",            r"\b(compare|comparison|vs\.?|versus|against|difference between)\b"),
    ("trend",              r"\b(when|trend|over time|evolution|evolve\w*|develop\w*|progress\w*|timeline|curve|daily|weekly|monthly|per (day|week|month)|time series|how did)\b"),
    ("top_n",              r"\b(top|highest|most|lowest|least|fewest|rank\w*|worst|best|biggest|largest|smallest|which (district|state)s?)\b"),
    ("profile",            r"\b(tell me about|profile|summary|overview|facts about|situation in)\b|(?<!what )(?<!how )\babout\b"),
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
    explicit: set[str] = set()
    from backend.explain import is_explain_request
    if is_explain_request(question):              # "what does this chart show?" is about the last answer
        p.intent = "explain"
        p.explicit = ["intent"]
        return p
    for intent, pattern in RULES:
        if re.search(pattern, q) or re.search(pattern, fq):
            p.intent = intent
            explicit.add("intent")
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
    # "which STATE shares the most borders with Niedersachsen" is about states, not districts
    if p.intent == "most_connected" and ex.states and not ex.districts and re.search(r"\bstates?\b", q):
        p.intent = "state_borders"
    if p.intent == "hotspot_trend" and ex.date and not (ex.start or ex.end):
        p.intent = "snapshot_map"

    metrics = _metric_mentions(fq)
    if metrics:
        explicit.add("metric")
    if p.intent == "correlation":
        p.metric = metrics[0] if metrics else "mobility"
        p.metric2 = metrics[1] if len(metrics) > 1 else ("incidence_7d" if p.metric != "incidence_7d" else "mobility")
    else:
        p.metric = metrics[0] if metrics else DEFAULT_METRIC.get(p.intent, "incidence_7d")
        if p.metric == "hotspot" and p.intent in ("trend",):
            p.intent = "hotspot_trend"

    p.level = "state" if re.search(LEVEL_STATE, fq) and not ex.districts else "district"
    if re.search(LEVEL_STATE, fq) and not ex.districts:
        explicit.add("level")
    p.order = "asc" if re.search(r"\b(lowest|least|fewest|smallest|best|minimum|bottom)\b", q) else "desc"
    if re.search(r"\b(lowest|least|fewest|smallest|best|minimum|bottom|highest|top|most|largest|worst)\b", q):
        explicit.add("order")
    p.peak = bool(re.search(r"\b(peak|maximum|max|highest ever)\b", q))
    if p.peak:
        explicit.add("peak")
    if re.search(r"\b(weekly|per week|by week|each week)\b", q):
        p.granularity = "week"
    elif re.search(r"\b(monthly|per month|by month|each month)\b", q):
        p.granularity = "month"
    elif re.search(r"\b(daily|per day|by day|each day)\b", q):
        p.granularity = "day"
    if re.search(r"\b(weekly|monthly|daily|per (week|month|day)|by (week|month|day)|each (week|month|day))\b", q):
        explicit.add("granularity")

    if p.intent == "compare" and (p.granularity != "day" or re.search(RULE["trend"], q)):
        p.intent = "trend"                   # "compare X and Y weekly / over time" -> multi-line trend
    t = re.search(r"\b(?:above|over|greater than|more than|exceed\w*|at least|>=?)\s*(\d+(?:\.\d+)?)", q)
    if t:
        p.threshold = float(t[1])
        explicit.add("threshold")

    m = re.search(r"\b(?:top|bottom|first|lowest|highest)\s+(\d{1,2})\b|\b(\d{1,2})\s+(?:districts|states|places)\b", q)
    if m:
        p.n = int(m[1] or m[2])
        explicit.add("n")
    h = re.search(r"\bwithin\s+(\d)\b|\b(\d)[- ]?(?:hops?|borders?|steps?)\b", q)
    if h:
        p.hops = int(h[1] or h[2])
        explicit.add("hops")
    elif re.search(r"neighbou?rs of neighbou?rs|second[- ]degree", q):
        p.hops = 2
        explicit.add("hops")

    p.districts = [{"id": int(d.id), "name": d.name} for d in ex.districts]
    p.states = [s.name for s in ex.states]
    p.date, p.start, p.end = ex.date, ex.start, ex.end
    p.notes = list(ex.notes)
    if ex.districts or ex.states:
        explicit.add("places")
    if ex.date or ex.start:
        explicit.add("dates")
    p.explicit = sorted(explicit)
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
Hotspot means 7-day incidence >= 75 per 100k.
If a "Previous plan" is given and the question is a follow-up to it, return ONLY the
fields the new question changes (omit the rest); never copy unchanged fields."""


def _llm_system() -> str:
    intents = "\n".join(f" {k}: {v[1]}" for k, v in INTENTS.items() if k != "explain")
    metrics = ", ".join(f"{k} ({v['label']})" for k, v in METRICS.items())
    return SYSTEM_PROMPT.format(first=config.FIRST_DATE, last=config.LAST_DATE,
                                intents=intents, metrics=metrics)


def _valid_date(s) -> str | None:
    if isinstance(s, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return min(max(s, config.FIRST_DATE), config.LAST_DATE)
    return None


def merge_llm(plan: QueryPlan, js: dict, linker: EntityLinker, question: str) -> QueryPlan:
    """Overlay validated LLM fields on the rule-based plan.

    Precedence rule: the LLM FILLS GAPS, it never overrules a slot the rules
    recognised with certainty (QueryPlan.explicit). The rules are tested and
    predictable; the LLM is used for what they could not read."""
    p = QueryPlan(**plan.to_dict())
    known = set(plan.explicit)
    if js.get("intent") in INTENTS and js.get("intent") != "explain" and "intent" not in known:
        p.intent = js["intent"]
    if "metric" not in known:
        if js.get("metric") in METRICS:
            p.metric = js["metric"]
        if js.get("metric2") in METRICS:
            p.metric2 = js["metric2"]
    if js.get("level") in ("district", "state") and "level" not in known:
        p.level = js["level"]
    if js.get("granularity") in ("day", "week", "month") and "granularity" not in known:
        p.granularity = js["granularity"]
    if js.get("order") in ("asc", "desc") and "order" not in known:
        p.order = js["order"]
    for key, lo, hi in (("n", 1, 50), ("hops", 1, 6)):
        if isinstance(js.get(key), int) and lo <= js[key] <= hi and key not in known:
            setattr(p, key, js[key])
    if isinstance(js.get("threshold"), (int, float)) and "threshold" not in known:
        p.threshold = float(js["threshold"])
    if isinstance(js.get("peak"), bool) and "peak" not in known:
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


SLOT_ATTRS = {"intent": ["intent"], "metric": ["metric", "metric2"], "level": ["level"],
              "order": ["order"], "n": ["n"], "hops": ["hops"], "granularity": ["granularity"],
              "threshold": ["threshold"], "peak": ["peak"], "dates": ["date", "start", "end"],
              "places": ["districts", "states"]}


def _previous_summary(last) -> str:
    """A compact (~100 token) view of the previous turn for the LLM; never the chat text."""
    p = last.plan
    summary = {k: p.get(k) for k in ("intent", "metric", "level", "date", "start", "end", "n", "hops")}
    summary["places"] = [d["name"] for d in p.get("districts", [])][:8] + list(p.get("states", []))
    return f"\nPrevious question: {last.question}\nPrevious plan: {json.dumps(summary, ensure_ascii=False)}"


def understand(question: str, linker: EntityLinker, last=None) -> tuple[QueryPlan, Extraction]:
    """Question -> QueryPlan. `last` is the previous turn of this conversation (memory.Turn) or None."""
    from backend.followup import is_followup, resolve

    ex = linker.extract(question)
    plan = rule_plan(question, ex)
    if plan.intent == "explain":                  # needs no new data, so no LLM parsing and no follow-up merge
        return plan, ex
    llm = get_llm()
    if llm.enabled:
        found = {"districts": [d.name for d in ex.districts], "states": [s.name for s in ex.states],
                 "date": ex.date, "start": ex.start, "end": ex.end}
        user = f"Question: {question}\nAlready detected: {json.dumps(found, ensure_ascii=False)}"
        if last is not None:
            user += _previous_summary(last)
        js = llm.complete_json(_llm_system(), user, max_tokens=600)
        if js:
            merged = merge_llm(plan, js, linker, question)
            # whatever the LLM added counts as stated by the user ...
            extra = {slot for slot, attrs in SLOT_ATTRS.items()
                     if any(getattr(merged, a) != getattr(plan, a) for a in attrs)}
            # ... except, inside a conversation, a question TYPE that only the LLM saw:
            # the user's words did not change it, so the remembered type is kept
            if last is not None:
                extra.discard("intent")
                if "intent" not in plan.explicit:
                    merged.intent = plan.intent
            merged.explicit = sorted(set(plan.explicit) | extra)
            plan = merged
    if is_followup(question, plan, last):
        plan, note = resolve(plan, last, question, QueryPlan,
                             linker.district_name, linker.district_state)
        plan.followup = note
    return finalize(plan), ex


def finalize(p: QueryPlan) -> QueryPlan:
    """Fill defaults and ask for clarification when a required place is missing."""
    # "when did Köln become a hotspot" -> the district's profile (first hotspot date + incidence curve)
    if p.intent == "hotspot_trend" and p.districts:
        p.intent = "profile"
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
    if p.intent == "subset" and not p.districts:
        p.intent = "top_n"                  # "subset" only makes sense with remembered districts
    if p.intent == "profile" and not (p.districts or p.states):
        p.intent = "top_n"
    if p.intent == "compare" and len(p.districts) + len(p.states) < 2:
        p.intent = "trend" if (p.districts or p.states) else "top_n"
    # a flow (new cases, deaths) with no date means "over the whole year", not "on the last day"
    if (p.intent in ("top_n", "compare", "neighbour_metrics", "k_hop", "snapshot_map")
            and METRICS.get(p.metric, {}).get("period") == "sum" and p.metric != "hotspot"
            and not (p.date or p.start)):
        p.start, p.end = config.FIRST_DATE, config.LAST_DATE
        p.notes.append("no date given for a cumulative measure, so the whole of 2020 is used")
    if p.metric == "hotspot" and p.intent in ("share", "snapshot_map", "compare"):
        p.metric = "incidence_7d"
    p.n = max(1, min(p.n, 50))
    p.hops = max(1, min(p.hops, 6))
    return p
