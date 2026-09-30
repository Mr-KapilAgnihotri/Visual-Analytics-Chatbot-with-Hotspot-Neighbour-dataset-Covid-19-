"""
The semantic layer: what the metrics and question types MEAN.

This one file is shared by the NLU (so the LLM knows what it may ask for),
the SQL engine (so it knows how to aggregate each metric) and the narrator
(so it can label things in plain English). Keeping meaning in one place is
what stops the three from drifting apart.
"""

# ---------------------------------------------------------------------------
# Metrics
#   col        : column in fact_daily
#   group_sql  : how to combine several districts on ONE day (state / national level)
#   period     : how to combine many days for one district
#                "sum"  -> flows (new cases, deaths): add them up
#                "last" -> stocks/levels (total cases, incidence): value on the last day
# ---------------------------------------------------------------------------
METRICS = {
    "new_cases": {
        "label": "New cases", "col": "new_cases",
        "group_sql": "SUM(new_cases)", "period": "sum",
        "synonyms": ["new cases", "daily cases", "infections", "new infections", "cases per day", "cases"],
    },
    "total_cases": {
        "label": "Total cases (cumulative)", "col": "total_cases",
        "group_sql": "SUM(total_cases)", "period": "last",
        "synonyms": ["total cases", "cumulative cases", "overall cases", "total infections"],
    },
    "cases_7d": {
        "label": "Cases in last 7 days", "col": "cases_7d",
        "group_sql": "SUM(cases_7d)", "period": "last",
        "synonyms": ["7 day cases", "seven day cases", "cases in the last 7 days"],
    },
    "incidence_7d": {
        "label": "7-day incidence per 100k", "col": "incidence_7d",
        # population-weighted: the correct way to combine incidences
        "group_sql": "SUM(cases_7d) * 100000.0 / SUM(population)", "period": "last",
        "synonyms": ["incidence", "7 day incidence", "infection rate", "rate", "per 100k"],
    },
    "deaths": {
        "label": "Deaths", "col": "deaths",
        "group_sql": "SUM(deaths)", "period": "sum",
        "synonyms": ["deaths", "fatalities", "died", "mortality"],
    },
    "hotspot": {
        "label": "Hotspot districts", "col": "hotspot",
        "group_sql": "SUM(CAST(hotspot AS INTEGER))", "period": "sum",   # per district: days as hotspot
        "synonyms": ["hotspot", "hotspots", "hot spot"],
    },
    "mobility": {
        "label": "Mobility change vs. baseline (%)", "state_level_only": True, "col": "mobility_value",
        "group_sql": "AVG(mobility_value)", "period": "avg",
        "synonyms": ["mobility", "movement", "travel"],
    },
    "population": {
        "label": "Population", "col": "population",
        "group_sql": "SUM(population)", "period": "last",
        "synonyms": ["population", "inhabitants", "people living"],
    },
}

# ---------------------------------------------------------------------------
# Intents (question types) and which engine answers them.
#   SQL    : aggregations over the 150k-row time series  -> DuckDB
#   GRAPH  : relationships / traversals over borders       -> Neo4j
#   HYBRID : graph finds WHO, SQL measures HOW MUCH         -> both, fused on district_id
# ---------------------------------------------------------------------------
INTENTS = {
    # ---- SQL ----
    "top_n":            ("SQL",    "Rank districts or states by a metric (on a date or over a period)"),
    "trend":            ("SQL",    "A metric over time for Germany, states or districts"),
    "share":            ("SQL",    "Share / breakdown of a metric by federal state"),
    "compare":          ("SQL",    "Compare a metric across named districts or states"),
    "correlation":      ("SQL",    "Relationship between two metrics (e.g. mobility vs incidence)"),
    "hotspot_trend":    ("SQL",    "Number of hotspot districts over time"),
    "snapshot_map":     ("SQL",    "Map of a metric for all districts on one date"),
    "profile":          ("SQL",    "Key figures for one district or state"),
    # ---- GRAPH ----
    "neighbours":       ("GRAPH",  "Which districts border a district"),
    "shortest_path":    ("GRAPH",  "Fewest border crossings between two districts"),
    "most_connected":   ("GRAPH",  "Districts with the most neighbours (degree centrality)"),
    "state_borders":    ("GRAPH",  "Which federal states border a state"),
    # ---- HYBRID ----
    "k_hop":            ("HYBRID", "Districts within k borders of a district, with their metric values"),
    "neighbour_metrics": ("HYBRID", "A district and its neighbours compared on a metric"),
    "hotspot_clusters": ("HYBRID", "Groups of bordering hotspot districts on a date"),
    "surrounded_hotspots": ("HYBRID", "Hotspots whose every neighbour is also a hotspot"),
    "isolated_hotspots": ("HYBRID", "Hotspots with no hotspot neighbour"),
    "help":             ("NONE",   "What the chatbot can do"),
}


def engine_for(intent: str) -> str:
    return INTENTS.get(intent, ("NONE", ""))[0]


def metric_label(metric: str) -> str:
    return METRICS.get(metric, {}).get("label", metric)
