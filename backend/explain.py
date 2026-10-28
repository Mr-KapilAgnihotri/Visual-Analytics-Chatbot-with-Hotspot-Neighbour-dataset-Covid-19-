"""
EXPLAIN THE PREVIOUS CHART: "what does this chart show?", "explain this map".

Such a question is about the conversation, not about the data: it needs no new
query and no new chart. The answer is built from what the previous turn
remembered (memory.Turn.answer): its question type, chart title, computed facts
and answer text, plus a fixed "how to read this chart" guide per question type.

    is_explain_request(question)  -> bool         (used by nlu.rule_plan)
    explain(question, last)       -> (text, facts) (used by the orchestrator)

The guide is written by hand, so it is always correct; the language model (if on)
may re-word guide + facts, and the usual number-grounding check still applies.
"""
from __future__ import annotations

import re

from backend import config

_ASK = r"(explain\w*|interpret\w*|meaning|mean|means|denote[sd]?|represent\w*|describe|understand|read|tell me about|show(s|n|ing)?|say(s|ing)?|telling|indicate\w*)"
_THING = r"(chart|graph|map|plot|visual\w*|figure|diagram|picture|image|graphic|answer|result|output)s?"
_POINTER = r"(this|that|the|above|previous|last|your|these|those)"
# "explain this leipzig chart", "what does the map denote", "how do I read this graph"
EXPLAIN_RE = re.compile(
    rf"\b{_ASK}\b.*\b{_POINTER}\s+(\w+\s+){{0,2}}{_THING}\b"
    rf"|\b{_POINTER}\s+(\w+\s+){{0,2}}{_THING}\b.*\b{_ASK}\b"
    rf"|^\s*(please\s+|can you\s+|could you\s+(please\s+)?)?(explain|interpret)(\s+(it|this|that)(\s+to me)?)?\s*(please)?[?.!]*\s*$"
    rf"|\bwhat does (it|this|that) (mean|show|denote|tell( me| us)?)\b"
    rf"|\bhow (do|should|can|to) (i |we )?read\b"
    rf"|\bwhat (does|do|is|are) .{{1,40}}\b(mean|means|stand for|stands for|denote|denotes)\b"
    rf"|\b(no|not|without|don'?t want|donot want|do not want)( an?| any)?( other| new| another)? (visual|chart|graph|plot)s?\b"
)


def is_explain_request(question: str) -> bool:
    q = question.lower()
    # "show me a map of incidence" is a NEW request, not a question about the last chart
    if re.match(r"^\s*(show|draw|plot|make|give|create)( me)?\b", q) and not re.search(r"\b(explain|mean|denote)", q):
        return False
    return bool(EXPLAIN_RE.search(q))


T = config.HOTSPOT_THRESHOLD
HOT = f"A district counts as a hotspot when its 7-day incidence is {T} or more new cases per 100,000 people."

# what the chart is, how to read it, and what to be careful about; {metric} is filled in
GUIDE = {
    "top_n": ("a horizontal bar chart (a ranking)",
              "Each bar is one district or state; the longer the bar, the higher its {metric}, and the number at the "
              "end of the bar is the exact value. The top bar is rank 1. When bars are red and grey, red means "
              "hotspot and grey means not a hotspot. " + HOT,
              "It shows a single day or period only; a district can rank high briefly because of one local outbreak."),
    "subset": ("a horizontal bar chart",
               "It shows the {metric} of exactly the districts from the previous answer, longest bar = highest value.",
               "The districts were chosen by the previous question, so this is not a ranking of all of Germany."),
    "trend": ("a line chart over time",
              "The horizontal axis is the date and the vertical axis is the {metric}; each line is one place "
              "(see the legend). A rising line means growth, a peak is the worst point. For incidence, the dashed "
              "line marks the hotspot threshold: " + HOT,
              "Daily values jump around (weekends report fewer cases); weekly or monthly views are smoother."),
    "compare": ("a bar chart comparing places",
                "Each bar is one place, so you can compare their {metric} side by side.",
                "Rates per 100,000 compare fairly between big and small places; raw counts favour big places."),
    "share": ("a share chart (donut or bars) by federal state",
              "The whole is Germany's total {metric}; each slice or bar is one federal state's part, labelled in percent.",
              "Big states have big shares simply because more people live there; look at incidence for a fair comparison."),
    "correlation": ("a scatter plot with a trend line",
                    "Each dot is one place (for mobility: one federal state, because mobility is recorded per state). "
                    "The horizontal axis is the first measure, the vertical axis the second. The orange line is the "
                    "best-fitting straight line. Pearson r runs from -1 to +1: the sign gives the direction, and the "
                    "closer to 0, the weaker the link (about 0.3 to 0.5 is moderate).",
                    "With only 16 states, one unusual state can change r a lot, and correlation does not prove causation."),
    "hotspot_trend": ("a line chart of the number of hotspot districts per day",
                      "The vertical axis counts how many districts were hotspots on each day. " + HOT,
                      "It counts districts, not people: a small rural district counts as much as a big city."),
    "snapshot_map": ("a map of Germany with one dot per district",
                     "Each dot sits at a district's centre and its colour shows the {metric}: darker means higher "
                     "(see the colour scale). Groups of dark dots show regional outbreaks.",
                     "Dots are centre points, not district outlines; small city districts can sit close together."),
    "profile": ("a district profile: key-number tiles and a line chart",
                "The tiles summarise the district (population, total cases, peak incidence, days as a hotspot, "
                "number of neighbours). The line is its 7-day incidence through 2020, and the dashed line is the "
                "hotspot threshold. " + HOT,
                "Small districts show sharper ups and downs, because a few cases change the rate a lot."),
    "neighbours": ("a network drawn on the map (from the knowledge graph)",
                   "Dots are districts at their centres and lines are shared borders. The highlighted dot is the "
                   "district you asked about; the other dots are the districts that border it.",
                   "A few borders around Berlin come from real geography, because the source data sums Berlin as one city."),
    "shortest_path": ("a path drawn on the map (from the knowledge graph)",
                      "The line follows shared borders from the start district to the end district; every dot on it is "
                      "a district you pass through. The number of borders crossed is the path length (hops).",
                      "It is the path with the fewest border crossings, not the shortest distance in kilometres."),
    "most_connected": ("a horizontal bar chart",
                       "Each bar is a district and its length is the number of districts it borders.",
                       "Districts that surround a city usually have many neighbours, which is geography, not COVID."),
    "state_borders": ("a horizontal bar chart",
                      "Each bar is a neighbouring federal state; its length is how many district-to-district borders "
                      "the two states share. The longest bar is the longest shared frontier.",
                      "It counts district borders, not kilometres of border."),
    "k_hop": ("a network drawn on the map",
              "The star is the start district; the other dots are districts within the chosen number of borders, "
              "coloured by {metric} (or by above/below the threshold you gave). Lines are shared borders.",
              "Further rings grow quickly, so a few hops already cover a large region."),
    "neighbour_metrics": ("a bar chart of a district and its neighbours",
                          "The highlighted bar is the district you asked about; the other bars are its neighbours, so "
                          "you see at once whether it is higher or lower than the area around it.",
                          "Neighbours can be cities or rural districts, which often differ for reasons other than COVID."),
    "hotspot_clusters": ("a map of hotspot clusters (from the knowledge graph)",
                         "Each colour is one cluster: hotspot districts joined by shared borders into one connected "
                         "region. Grey dots are small clusters or single hotspots; faint lines are all district borders. " + HOT,
                         "A cluster can become huge when many neighbouring districts cross the threshold at the same time."),
    "surrounded_hotspots": ("a map of hotspot districts",
                            "Red dots are hotspots whose every neighbour is also a hotspot; orange dots are other hotspots; "
                            "grey dots are not hotspots. Many red dots mean regional, not local, spread. " + HOT,
                            "A district on the national border has fewer neighbours in the data, so it is 'surrounded' more easily."),
    "isolated_hotspots": ("a map of hotspot districts",
                          "Red dots are hotspots with no hotspot neighbour at all (local outbreaks); orange dots are "
                          "other hotspots; grey dots are not hotspots. " + HOT,
                          "An isolated hotspot often points to one local event, such as an outbreak in one facility."),
}

NO_CHART = ("There is no chart yet to explain. Ask a question first, for example "
            "\"Top 10 districts by incidence on 15 November\", then ask \"what does this chart show?\".")


def explain(question: str, last) -> tuple[str, dict]:
    """Text-only explanation of the previous turn's chart. `last` is a memory.Turn or None."""
    from backend.catalog import metric_label

    prev = getattr(last, "answer", None) or {}
    if not prev.get("intent") or prev["intent"] not in GUIDE:
        return NO_CHART, {}
    kind, how, caveat = GUIDE[prev["intent"]]
    metric = metric_label(prev.get("metric")) if prev.get("metric") else "value"
    how = how.format(metric=metric.lower() if metric[:1].isupper() and not metric[:2].isupper() else metric)
    title = prev.get("chart_title") or ""
    text = (f"**What the chart shows.** It is {kind}"
            + (f": \"{title}\"" if title else "") + f", answering \"{last.question}\".\n\n"
            f"**How to read it.** {how}\n\n"
            f"**What it tells us.** {prev.get('text', '').strip()}\n\n"
            f"**Keep in mind.** {caveat}")
    facts = {"explaining_question": last.question, "chart_type": kind, "chart_title": title,
             "how_to_read": how, "caveat": caveat, "previous_answer": prev.get("text", ""),
             "previous_facts": prev.get("facts", {})}
    return text, facts
