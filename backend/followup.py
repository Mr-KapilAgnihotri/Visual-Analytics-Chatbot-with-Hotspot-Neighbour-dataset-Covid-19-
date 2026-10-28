"""
FOLLOW-UP RESOLUTION: turning a short follow-up into a complete question.

Theory (dialogue-state tracking with slot carry-over)
-----------------------------------------------------
The "dialogue state" is the previous turn's QueryPlan. A follow-up question
usually states only what CHANGES; everything it leaves out is inherited:

    turn 1  "Top 10 districts by incidence on 15 November"  -> top_n, incidence_7d, n=10, 2020-11-15
    turn 2  "And on 1 December?"                             -> states only the date
            resolved:                                           top_n, incidence_7d, n=10, 2020-12-01

The kinds of follow-up handled (linguistic names in brackets):
  * leaving words out:   "and on 1 December?", "weekly", "top 5"        (ellipsis)
  * pointing back:       "their trend", "its neighbours"               (anaphora / coreference)
  * swapping a place:    "what about Hamburg?", "Hamburg instead"      (substitution)
  * adding a place:      "also add Hessen"                             (extension)
  * narrowing:           "only the ones in Bavaria"                    (refinement)
  * using the ANSWER:    "which of them are hotspots?"                 (result reference)

Two rules make it work:
  1. a slot the new question STATES (QueryPlan.explicit) always wins;
  2. a slot it does not state is copied from the previous plan.
Everything here is deterministic (no LLM needed), and the resolved plan still
passes through nlu.finalize(), so every existing validation still applies.
"""
from __future__ import annotations

import copy
import re
from datetime import date as _date

from backend.catalog import metric_label

LINK = re.compile(r"^\s*(and|also|plus|but|so|now|then|ok(ay)?|what about|how about|same|instead|only|just)\b")
POINT_PLURAL = re.compile(r"\b(their|them|they|those|these|the ones|those districts|these districts|the same districts)\b")
POINT_SINGLE = re.compile(r"\b(its|it|that district|this district|that one|this one|the same district)\b")
ADD = re.compile(r"\b(also|add|plus|too|as well|together with)\b")
ONLY = re.compile(r"\b(only|just|restricted to|limited to|limit to)\b")

# question types that act on named places, so a bare "show the trend" can mean "for the same places"
PLACE_INTENTS = {"trend", "compare", "profile", "neighbours", "k_hop", "neighbour_metrics",
                 "shortest_path", "state_borders", "subset"}
SLOT_FIELDS = {"intent": ["intent"], "metric": ["metric", "metric2"], "level": ["level"], "order": ["order"],
               "peak": ["peak"], "granularity": ["granularity"], "n": ["n"], "hops": ["hops"],
               "threshold": ["threshold"], "dates": ["date", "start", "end"]}
FRIENDLY_INTENT = {
    "top_n": "ranking", "trend": "trend over time", "share": "share by state", "compare": "comparison",
    "correlation": "correlation", "hotspot_trend": "hotspot count over time", "snapshot_map": "map",
    "profile": "profile", "neighbours": "neighbours", "shortest_path": "shortest path",
    "most_connected": "most connected districts", "state_borders": "neighbouring states",
    "k_hop": "districts within k borders", "neighbour_metrics": "comparison with neighbours",
    "hotspot_clusters": "hotspot clusters", "surrounded_hotspots": "surrounded hotspots",
    "isolated_hotspots": "isolated hotspots", "subset": "values for the previous answer's districts",
}
MAX_TREND_LINES = 8
# question types that do not use a measure at all
METRIC_FREE = {"neighbours", "shortest_path", "most_connected", "state_borders",
               "hotspot_clusters", "surrounded_hotspots", "isolated_hotspots", "profile"}
# question types whose answer does not depend on named places (don't claim we "kept" them)
PLACE_FREE = {"snapshot_map", "hotspot_clusters", "surrounded_hotspots", "isolated_hotspots",
              "correlation", "share", "most_connected"}


def _words(text: str) -> int:
    return len(re.findall(r"\w+", text))


def is_followup(question: str, plan, last) -> bool:
    """Decide whether `question` continues the previous turn (`last`) or starts a new topic."""
    if last is None:
        return False
    q = question.lower().strip()
    explicit = set(plan.explicit)
    if LINK.search(q):                                         # "and ...", "what about ...", "only ..."
        return True
    if (POINT_PLURAL.search(q) or POINT_SINGLE.search(q)) and not {"intent", "places"} <= explicit:
        return True                                            # "their trend", "its neighbours"
    if "intent" not in explicit and explicit and _words(q) <= 8:
        return True                                            # "Hamburg?", "on 1 December", "weekly"
    had_places = bool(last.plan.get("districts") or last.plan.get("states") or last.result_ids)
    if ("intent" in explicit and "places" not in explicit and plan.intent in PLACE_INTENTS
            and had_places and _words(q) <= 5):
        return True                                            # "show the trend", "neighbours?"
    return False


def _describe_dates(p: dict) -> str:
    def d(s):
        x = _date.fromisoformat(str(s)[:10])
        return f"{x.day} {x.strftime('%b %Y')}"
    if p.get("start") and p.get("end"):
        return f"{d(p['start'])} to {d(p['end'])}"
    if p.get("date"):
        return d(p["date"])
    return "default dates"


def _describe_places(p: dict) -> str:
    names = [x["name"] for x in p.get("districts", [])] + list(p.get("states", []))
    if not names:
        return ""
    return ", ".join(names[:3]) + (f" and {len(names) - 3} more" if len(names) > 3 else "")


def resolve(new, last, question: str, make_plan, district_name: dict, district_state: dict):
    """Merge the new (partial) plan with the previous turn. Returns (plan, note)."""
    q = question.lower()
    prev = copy.deepcopy(last.plan)
    explicit = set(new.explicit)
    base = dict(prev)
    base.update(nlu_source=new.nlu_source, notes=list(new.notes), clarification=None,
                followup=None, explicit=list(new.explicit))
    changed: list[str] = []

    # 1) every stated slot wins
    for slot, fields in SLOT_FIELDS.items():
        if slot in explicit:
            for f in fields:
                base[f] = getattr(new, f)
    if base["intent"] != prev["intent"]:
        changed.append(FRIENDLY_INTENT.get(base["intent"], base["intent"]))
        # a single remembered day makes no sense for a trend: show the whole year instead
        if base["intent"] in ("trend", "hotspot_trend") and "dates" not in explicit and prev.get("date"):
            base["date"] = None
    if "metric" in explicit and new.metric != prev.get("metric") and not POINT_PLURAL.search(q):
        changed.append(metric_label(new.metric))
    if "dates" in explicit and (new.date, new.start, new.end) != (prev.get("date"), prev.get("start"), prev.get("end")):
        changed.append(_describe_dates(base))
    for slot, word in (("n", "top {}"), ("hops", "{} borders"), ("granularity", "{}"), ("threshold", "above {:g}")):
        if slot in explicit and getattr(new, slot) != prev.get(slot):
            changed.append(word.format(getattr(new, slot)))

    # 2) places: replace, add, narrow, or point back to the previous answer
    new_d = [dict(x) for x in new.districts]
    new_s = list(new.states)
    plural = POINT_PLURAL.search(q)
    result_districts = [{"id": i, "name": district_name.get(i, str(i))} for i in last.result_ids]
    if "places" in explicit:
        if ONLY.search(q) and (prev.get("districts") or result_districts) and new_s and not new_d:
            if base["intent"] in ("top_n",):
                base["districts"], base["states"] = [], new_s          # top_n filters by state itself
            else:
                pool = prev.get("districts") or result_districts
                kept = [x for x in pool if district_state.get(x["id"]) in new_s]
                if kept:
                    base["districts"], base["states"] = kept, []
                else:
                    base["districts"], base["states"] = [], new_s
            changed.append("only " + ", ".join(new_s))
        elif ADD.search(q):
            seen = {x["id"] for x in prev.get("districts", [])}
            base["districts"] = prev.get("districts", []) + [x for x in new_d if x["id"] not in seen]
            base["states"] = prev.get("states", []) + [s for s in new_s if s not in prev.get("states", [])]
            changed.append("added " + ", ".join([x["name"] for x in new_d] + new_s))
        else:
            if base["intent"] == "shortest_path" and len(new_d) == 1 and prev.get("districts"):
                new_d = prev["districts"][:1] + new_d                   # "and to Passau?" keeps the start
            same = ({x["id"] for x in new_d}, set(new_s)) == ({x["id"] for x in prev.get("districts", [])},
                                                              set(prev.get("states", [])))
            base["districts"], base["states"] = new_d, new_s
            if not same:
                changed.append(_describe_places(base))
    elif plural and (result_districts or last.result_states):
        if result_districts:
            base["districts"], base["states"] = result_districts, []
        else:
            base["districts"], base["states"] = [], list(last.result_states)
        # "which of them are hotspots?" / "their deaths" after a ranking -> values for exactly those places
        if "intent" not in explicit and prev["intent"] not in ("trend", "compare", "subset") and result_districts:
            base["intent"] = "subset"
            if base.get("metric") == "hotspot" or re.search(r"\bhotspots?\b", q):
                base["metric"] = "incidence_7d"
            changed.append(FRIENDLY_INTENT["subset"])
        changed.append(f"the {len(base['districts']) or len(base['states'])} places from the previous answer")

    # 3) keep charts readable
    if base["intent"] == "trend" and len(base.get("districts", [])) > MAX_TREND_LINES:
        base["districts"] = base["districts"][:MAX_TREND_LINES]
        changed.append(f"first {MAX_TREND_LINES} districts (one line each)")
    if base["intent"] == "subset":
        base["districts"] = base.get("districts", [])[:50]

    plan = make_plan(**base)

    kept = []
    if base["intent"] == prev["intent"]:
        kept.append(FRIENDLY_INTENT.get(base["intent"], base["intent"]))
    if "metric" not in explicit and base.get("metric") and base["intent"] not in METRIC_FREE:
        kept.append(metric_label(base["metric"]))
    if ("places" not in explicit and not plural and _describe_places(base)
            and base["intent"] not in PLACE_FREE):
        kept.append(_describe_places(base))
    if "dates" not in explicit and (base.get("date") or base.get("start")):
        kept.append(_describe_dates(base))
    if base["intent"] == "top_n" and "n" not in explicit:
        kept.append(f"top {base['n']}")
    note = f'Follow-up to "{last.question}"'
    if changed:
        note += ": changed " + ", ".join(dict.fromkeys(changed))
    if kept:
        note += "; kept " + ", ".join(dict.fromkeys(kept))
    return plan, note
