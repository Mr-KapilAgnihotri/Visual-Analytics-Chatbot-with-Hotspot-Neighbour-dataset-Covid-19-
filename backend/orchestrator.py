"""
The ORCHESTRATOR: the pipeline that turns one question into one answer.

    question
      -> [1] understand   (nlu.py)            QueryPlan: intent, metric, places, dates ...
      -> [2] route        (catalog.INTENTS)   SQL | GRAPH | HYBRID
      -> [3] execute      (sql_engine / graph_engine)
      -> [4] fuse         (pandas merge on district_id, for HYBRID)
      -> [5] visualise    (charts.py)
      -> [6] narrate      (narrator.py)       text grounded in computed facts
      -> Answer (text + chart + table + the exact queries and timings)

Each intent has one handler method below. A handler returns
(figure, facts, fallback_text, table, kpis).
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from datetime import date as _date

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from backend import charts, config
from backend.catalog import INTENTS, METRICS, engine_for, metric_label
from backend.charts import CONTEXT, FOCUS, HOTSPOT, SERIES, fmt
from backend.entities import EntityLinker
from backend.graph_engine import GraphResult, get_graph, graph_status
from backend.narrator import narrate
from backend.nlu import QueryPlan, understand
from backend.sql_engine import SqlResult, get_sql

HELP_TEXT = """I answer questions about COVID-19 in Germany's 411 districts during 2020.
Relational questions (DuckDB): "Top 10 districts by incidence on 15 November", "Share of cases by federal state",
"How did new cases develop in Bavaria?", "Does mobility correlate with incidence in the second wave?".
Graph questions (Neo4j): "Which districts border Köln?", "Shortest path from Flensburg to Passau",
"Most connected districts", "Which states border Hessen?".
Hybrid questions (both): "Districts within 2 borders of Berlin Mitte with incidence above 100",
"Compare Köln with its neighbours", "Clusters of connected hotspots on 1 December",
"Hotspots surrounded by hotspots on 1 December"."""


@dataclass
class Answer:
    question: str
    intent: str
    engine: str
    text: str
    narrative_source: str = "template"
    nlu_source: str = "rules"
    chart: str | None = None
    table: list = field(default_factory=list)
    kpis: list = field(default_factory=list)
    queries: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    facts: dict = field(default_factory=dict)
    plan: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    graph_backend: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _d(s: str | None) -> str:
    if not s:
        return ""
    d = _date.fromisoformat(str(s)[:10])
    return f"{d.day} {d.strftime('%b %Y')}"


class Orchestrator:
    def __init__(self):
        self.sql = get_sql()
        self.graph = get_graph()
        self.districts = self.sql.districts()                # 411 rows, cached for coordinates
        self.linker = EntityLinker(self.districts)
        self._all_edges: list | None = None

    # ------------------------------------------------------------------
    def ask(self, question: str) -> Answer:
        t0 = time.perf_counter()
        self._queries: list[dict] = []
        plan, _ = understand(question, self.linker)
        t_nlu = (time.perf_counter() - t0) * 1000

        ans = Answer(question=question, intent=plan.intent, engine=engine_for(plan.intent), text="",
                     nlu_source=plan.nlu_source, plan=plan.to_dict(), notes=list(plan.notes),
                     graph_backend=self.graph.backend)
        if plan.clarification:
            ans.text, ans.engine = plan.clarification, "NONE"
            ans.timings = {"nlu_ms": round(t_nlu, 1), "total_ms": round((time.perf_counter() - t0) * 1000, 1)}
            return ans
        if plan.intent == "help":
            ans.text, ans.engine = HELP_TEXT, "NONE"
            return ans

        t1 = time.perf_counter()
        handler = getattr(self, f"_h_{plan.intent}")
        fig, facts, fallback, table, kpis = handler(plan)
        t_data = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        chart_desc = fig.layout.title.text if fig is not None and fig.layout.title.text else ""
        ans.text, ans.narrative_source = narrate(question, chart_desc, facts, fallback)
        t_narr = (time.perf_counter() - t2) * 1000

        ans.chart = charts.to_json(fig)
        ans.facts, ans.table, ans.kpis = facts, table[:60], kpis
        ans.queries = self._queries
        used = {"SQL" if q["engine"] == "DuckDB" else "GRAPH" for q in self._queries}
        if used:                                   # report the engines that were ACTUALLY used
            ans.engine = "HYBRID" if len(used) == 2 else used.pop()
        ans.timings = {
            "nlu_ms": round(t_nlu, 1),
            "sql_ms": round(sum(q["ms"] for q in self._queries if q["engine"] == "DuckDB"), 1),
            "graph_ms": round(sum(q["ms"] for q in self._queries if q["engine"] != "DuckDB"), 1),
            "build_ms": round(t_data, 1),          # queries + fusion + chart
            "narrate_ms": round(t_narr, 1),
            "total_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
        return ans

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _log(self, r: SqlResult | GraphResult, purpose: str) -> None:
        if isinstance(r, SqlResult):
            self._queries.append({"engine": "DuckDB", "language": "SQL", "purpose": purpose,
                                  "text": r.query, "ms": round(r.ms, 2), "rows": int(len(r.df))})
        else:
            eng = "Neo4j" if self.graph.backend == "neo4j" else "NetworkX"
            self._queries.append({"engine": eng, "language": r.language, "purpose": purpose,
                                  "text": r.query, "ms": round(r.ms, 2), "rows": len(r.rows)})

    def _sql(self, r: SqlResult, purpose: str) -> pd.DataFrame:
        self._log(r, purpose)
        return r.df

    def _g(self, r: GraphResult, purpose: str) -> GraphResult:
        self._log(r, purpose)
        return r

    def _period(self, p: QueryPlan) -> str:
        if p.start and p.end:
            if p.start == config.FIRST_DATE and p.end == config.LAST_DATE:
                return "over 2020"
            return f"from {_d(p.start)} to {_d(p.end)}"
        return f"on {_d(p.date or config.LAST_DATE)}"

    def _snapshot_desc(self, p: QueryPlan, metric: str) -> str:
        """For level metrics a period collapses to its last day - say so honestly."""
        m = METRICS[metric]
        if p.start and p.end and m["period"] == "last":
            return f"on {_d(p.end)} (end of the period)"
        return self._period(p)

    def _coords(self, ids) -> pd.DataFrame:
        return self.districts[self.districts["district_id"].isin(list(ids))].copy()

    def _all_edge_pairs(self) -> list:
        if self._all_edges is None:
            self._all_edges = [(r["a"], r["b"]) for r in self.graph.all_edges().rows]
        return self._all_edges

    def _snapshot_date(self, p: QueryPlan) -> str:
        return p.date or p.end or config.LAST_DATE

    @staticmethod
    def _records(df: pd.DataFrame) -> list:
        out = df.copy()
        for c in out.columns:
            if pd.api.types.is_datetime64_any_dtype(out[c]):
                out[c] = out[c].dt.strftime("%Y-%m-%d")
            elif pd.api.types.is_float_dtype(out[c]):
                out[c] = out[c].round(2)
        out = out.astype(object).where(pd.notna(out), None)
        return out.to_dict("records")

    # ==================================================================
    # SQL handlers
    # ==================================================================
    def _h_top_n(self, p: QueryPlan):
        lab = metric_label(p.metric)
        states = p.states if p.level == "district" else None
        df = self._sql(self.sql.top_n(p.metric, p.level, p.date, p.start, p.end, p.n, p.order, states, p.peak),
                       f"rank {p.level}s by {lab}")
        if df.empty:
            return None, {}, "No data matched that question.", [], []
        when = ("peak " if p.peak and p.start else "") + self._snapshot_desc(p, p.metric)
        where = f" in {', '.join(p.states)}" if states else ""
        word = "Lowest" if p.order == "asc" else "Top"
        title = f"{word} {len(df)} {'states' if p.level == 'state' else 'districts'}{where} by {lab.lower()}, {when}"
        hotspot_mode = p.metric == "incidence_7d" and p.level == "district"
        fig = charts.bar_ranking(df, title, lab, "hotspot" if hotspot_mode else "focus")
        rows = df.to_dict("records")
        facts = {"metric": lab, "period": when, "order": "lowest first" if p.order == "asc" else "highest first",
                 "ranking": [{"rank": i + 1, "name": r["name"], "state": r.get("state"), "value": round(float(r["value"]), 1)}
                             for i, r in enumerate(rows)]}
        if len(rows) > 1:
            facts["ratio_first_to_last_shown"] = round(float(rows[0]["value"]) / max(float(rows[-1]["value"]), 1e-9), 2)
        if hotspot_mode:
            facts["hotspots_among_shown"] = int((df["value"] >= config.HOTSPOT_THRESHOLD).sum())
        if p.level == "district" and "state" in df:
            facts["states_represented"] = df["state"].value_counts().to_dict()
        r0 = rows[0]
        lead = "has the lowest value" if p.order == "asc" else "ranks first"
        text = (f"{r0['name']} {lead} with {fmt(r0['value'])} ({lab.lower()}, {when})")
        if len(rows) > 2:
            text += f", followed by {rows[1]['name']} ({fmt(rows[1]['value'])}) and {rows[2]['name']} ({fmt(rows[2]['value'])})"
        text += "."
        if hotspot_mode:
            text += f" {facts['hotspots_among_shown']} of the {len(rows)} districts shown are hotspots (≥ 75 per 100,000)."
        return fig, facts, text, self._records(df), []

    def _h_trend(self, p: QueryPlan):
        lab = metric_label(p.metric)
        ids = [d["id"] for d in p.districts]
        start, end = (p.start, p.end) if p.start else (config.FIRST_DATE, config.LAST_DATE)
        if p.date and not p.start:
            start, end = config.FIRST_DATE, p.date
        gran = p.granularity
        df = self._sql(self.sql.trend(p.metric, ids, p.states, start, end, gran), f"{lab} over time")
        if df.empty:
            return None, {}, "No data for that period.", [], []
        places = ", ".join([d["name"] for d in p.districts] + p.states) or "Germany"
        per = {"day": "daily", "week": "weekly", "month": "monthly"}[gran]
        title = f"{lab} ({per}) – {places}, {_d(start)} to {_d(end)}"
        fig = charts.lines(df, title, lab, config.HOTSPOT_THRESHOLD if p.metric == "incidence_7d" else None)
        facts = {"metric": lab, "granularity": per, "period": f"{_d(start)} to {_d(end)}", "series": []}
        texts = []
        for s, g in df.groupby("series"):
            g = g.sort_values("period")
            pk = g.loc[g["value"].idxmax()]
            item = {"name": s, "first_value": round(float(g["value"].iloc[0]), 1),
                    "last_value": round(float(g["value"].iloc[-1]), 1),
                    "peak_value": round(float(pk["value"]), 1), "peak_period": str(pk["period"])[:10]}
            if METRICS[p.metric]["period"] == "sum":
                item["total_over_period"] = round(float(g["value"].sum()), 0)
            facts["series"].append(item)
            when_pk = (pd.Timestamp(pk["period"]).strftime("in %b %Y") if gran == "month"
                       else f"in the week of {_d(pk['period'])}" if gran == "week" else f"on {_d(pk['period'])}")
            item["peak_period"] = when_pk
            texts.append(f"{s} peaked at {fmt(pk['value'])} {when_pk} and ended the period at "
                         f"{fmt(g['value'].iloc[-1])}")
        text = f"{lab} ({per}), {_d(start)} to {_d(end)}: " + "; ".join(texts) + "."
        return fig, facts, text, self._records(df.tail(400)), []

    def _h_share(self, p: QueryPlan):
        lab = metric_label(p.metric)
        m = METRICS[p.metric]
        df = self._sql(self.sql.share_by_state(p.metric, p.date, p.start, p.end), f"{lab} by state")
        when = self._period(p) if m["period"] == "sum" and (p.start or p.date) else (
            "over 2020" if m["period"] == "sum" else self._snapshot_desc(p, p.metric))
        if m["period"] == "sum" and not (p.start or p.date):
            when = "over 2020"
        is_rate = p.metric in ("incidence_7d", "mobility")
        if is_rate:
            fig = charts.bar_ranking(df, f"{lab} by federal state, {when}", lab)
            note = "A rate cannot be split into shares of a whole, so it is shown as a ranking instead of a pie."
        else:
            fig = charts.donut(df, f"Share of {lab.lower()} by federal state, {when}", lab)
            note = None
        total = float(df["value"].sum())
        df["share_pct"] = df["value"] / total * 100 if total else 0
        top3 = df.head(3)
        facts = {"metric": lab, "period": when, "total": round(total), "states": len(df),
                 "top": [{"state": r["name"], "value": round(float(r["value"])), "share_pct": round(float(r["share_pct"]), 1)}
                         for _, r in df.head(6).iterrows()],
                 "top3_combined_share_pct": round(float(top3["share_pct"].sum()), 1)}
        if note:
            facts["note"] = note
        text = (f"{top3.iloc[0]['name']} accounts for the largest share of {lab.lower()} {when} "
                f"({top3.iloc[0]['share_pct']:.1f}%), and the three largest states together make up "
                f"{facts['top3_combined_share_pct']}% of the national total of {fmt(total)}.")
        if is_rate:
            text = f"{df.iloc[0]['name']} has the highest {lab.lower()} {when} ({fmt(df.iloc[0]['value'])}). {note}"
        return fig, facts, text, self._records(df), []

    def _h_compare(self, p: QueryPlan):
        lab = metric_label(p.metric)
        df = self._sql(self.sql.compare(p.metric, [d["id"] for d in p.districts], p.states, p.date, p.start, p.end),
                       f"compare {lab}")
        when = self._snapshot_desc(p, p.metric)
        fig = charts.bar_ranking(df, f"{lab}: comparison, {when}", lab,
                                 "hotspot" if p.metric == "incidence_7d" else "focus")
        hi, lo = df.iloc[0], df.iloc[-1]
        ratio = float(hi["value"]) / float(lo["value"]) if float(lo["value"]) else None
        facts = {"metric": lab, "period": when,
                 "values": [{"name": r["name"], "kind": r["kind"], "value": round(float(r["value"]), 1)} for _, r in df.iterrows()],
                 "highest": hi["name"], "lowest": lo["name"], "ratio_highest_to_lowest": round(ratio, 2) if ratio else None}
        text = f"{hi['name']} has the higher {lab.lower()} {when} ({fmt(hi['value'])}) versus {lo['name']} ({fmt(lo['value'])})"
        text += f", about {ratio:.1f} times as much." if ratio else "."
        return fig, facts, text, self._records(df), []

    def _h_correlation(self, p: QueryPlan):
        x, y = p.metric, p.metric2 or "incidence_7d"
        lx, ly = metric_label(x), metric_label(y)
        state_level = METRICS[x].get("state_level_only") or METRICS[y].get("state_level_only") or p.level == "state"
        start, end = (p.start, p.end) if p.start else ((None, None) if p.date else (config.FIRST_DATE, config.LAST_DATE))
        df = self._sql(self.sql.correlation(x, y, p.date, start, end, "state" if state_level else "district"),
                       f"{lx} vs {ly}")
        when = self._period(QueryPlan(date=p.date, start=start, end=end))
        unit = "federal states" if state_level else "districts"
        fig, st = charts.scatter(df, f"{ly} vs {lx}, {unit}, {when}", lx, ly,
                                 "Federal state" if state_level else "District")
        r = st["r"]
        strength = ("no meaningful" if r is None or abs(r) < 0.1 else "a weak" if abs(r) < 0.3
                    else "a moderate" if abs(r) < 0.6 else "a strong")
        direction = "" if r is None or abs(r) < 0.1 else (" positive" if r > 0 else " negative")
        facts = {"x": lx, "y": ly, "period": when, "points": st["n"], "unit": unit,
                 "pearson_r": round(r, 3) if r is not None else None,
                 "strength": f"{strength}{direction} linear relationship",
                 "caveat": "Correlation does not prove causation; other factors (season, measures, testing) also vary."}
        if state_level and (METRICS[x].get("state_level_only") or METRICS[y].get("state_level_only")):
            facts["data_note"] = ("Mobility is recorded per federal state, so each point is a state; "
                                  "using districts would repeat the same mobility value many times.")
        text = (f"Across {st['n']} {unit} ({when}) there is {strength}{direction} linear relationship between "
                f"{lx.lower()} and {ly.lower()} (Pearson r = {fmt(r) if r is not None else 'n/a'}). "
                f"{facts['caveat']}")
        return fig, facts, text, self._records(df), []

    def _h_hotspot_trend(self, p: QueryPlan):
        start, end = (p.start, p.end) if p.start else (config.FIRST_DATE, config.LAST_DATE)
        df = self._sql(self.sql.hotspot_trend(p.states, start, end), "hotspot districts per day")
        places = ", ".join(p.states) or "Germany"
        fig = charts.lines(df, f"Number of hotspot districts – {places}, {_d(start)} to {_d(end)}", "Hotspot districts")
        facts = {"period": f"{_d(start)} to {_d(end)}", "series": []}
        parts = []
        for s, g in df.groupby("series"):
            pk = g.loc[g["value"].idxmax()]
            first = g[g["value"] > 0]
            facts["series"].append({"name": s, "peak_count": int(pk["value"]), "peak_date": str(pk["period"])[:10],
                                    "count_on_last_day": int(g["value"].iloc[-1]),
                                    "first_day_with_hotspot": str(first["period"].iloc[0])[:10] if len(first) else None})
            parts.append(f"{s} reached a maximum of {int(pk['value'])} hotspot districts on {_d(pk['period'])}")
        text = "; ".join(parts) + "."
        return fig, facts, text, self._records(df), []

    def _h_snapshot_map(self, p: QueryPlan):
        metric = p.metric if p.metric != "hotspot" else "incidence_7d"
        lab = metric_label(metric)
        df = self._sql(self.sql.snapshot(metric, p.date, None, p.start if METRICS[metric]["period"] != "last" else None,
                                         p.end if METRICS[metric]["period"] != "last" else None),
                       f"{lab} for every district")
        if METRICS[metric]["period"] == "last" and p.end and not p.date:
            p.date = p.end
        when = self._snapshot_desc(p, metric)
        fig = charts.value_map(df, f"{lab} by district, {when}", lab)
        top = df.nlargest(5, "value")
        facts = {"metric": lab, "period": when, "districts": len(df),
                 "median": round(float(df["value"].median()), 1), "max": round(float(df["value"].max()), 1),
                 "top5": [{"name": r["name"], "state": r["state"], "value": round(float(r["value"]), 1)} for _, r in top.iterrows()]}
        if metric == "incidence_7d":
            facts["hotspots"] = int((df["value"] >= config.HOTSPOT_THRESHOLD).sum())
            by_state = df.assign(h=df["value"] >= config.HOTSPOT_THRESHOLD).groupby("state")["h"].mean()
            facts["state_with_highest_hotspot_share"] = by_state.idxmax()
        text = (f"The map shows {lab.lower()} for all {len(df)} districts {when}. The highest value is in "
                f"{top.iloc[0]['name']} ({fmt(top.iloc[0]['value'])}); the median district has {fmt(facts['median'])}.")
        if "hotspots" in facts:
            text += f" {facts['hotspots']} districts are hotspots."
        return fig, facts, text, self._records(df.sort_values("value", ascending=False)), []

    def _h_profile(self, p: QueryPlan):
        if p.districts:
            d = p.districts[0]
            prof = self._sql(self.sql.district_profile(d["id"]), "district key figures").iloc[0]
            nb = self._g(self.graph.neighbours(d["id"]), "neighbouring districts")
            trend = self._sql(self.sql.trend("incidence_7d", [d["id"]], None), "incidence over time")
            name = d["name"]
            kpis = [{"label": "Population", "value": f"{int(prof['population']):,}"},
                    {"label": "Total cases (31 Dec)", "value": f"{int(prof['total_cases']):,}"},
                    {"label": "Peak 7-day incidence", "value": f"{prof['peak_incidence']:.0f}"},
                    {"label": "Days as hotspot", "value": f"{int(prof['days_as_hotspot'])}"},
                    {"label": "Neighbouring districts", "value": str(len(nb.rows))}]
            facts = {"district": name, "state": prof["state"], "population": int(prof["population"]),
                     "total_cases_end_2020": int(prof["total_cases"]), "deaths_2020": int(prof["deaths"]),
                     "peak_incidence": round(float(prof["peak_incidence"]), 1), "peak_date": str(prof["peak_date"])[:10],
                     "days_as_hotspot": int(prof["days_as_hotspot"]),
                     "first_hotspot_date": str(prof["first_hotspot"])[:10] if pd.notna(prof["first_hotspot"]) else None,
                     "incidence_on_31_dec": round(float(prof["incidence_last_day"]), 1),
                     "neighbours": [r["name"] for r in nb.rows]}
            text = (f"{name} ({prof['state']}, population {int(prof['population']):,}) recorded "
                    f"{int(prof['total_cases']):,} cases in 2020. Its 7-day incidence peaked at "
                    f"{prof['peak_incidence']:.0f} on {_d(prof['peak_date'])}, and it was a hotspot on "
                    f"{int(prof['days_as_hotspot'])} days. It borders {len(nb.rows)} districts.")
        else:
            st = p.states[0]
            prof = self._sql(self.sql.state_profile(st), "state key figures").iloc[0]
            sb = self._g(self.graph.state_borders(st), "neighbouring states")
            trend = self._sql(self.sql.trend("incidence_7d", None, [st]), "incidence over time")
            name = st
            kpis = [{"label": "Population", "value": f"{int(prof['population']):,}"},
                    {"label": "Districts", "value": str(int(prof["n_districts"]))},
                    {"label": "Total cases (31 Dec)", "value": f"{int(prof['total_cases']):,}"},
                    {"label": "Peak 7-day incidence", "value": f"{prof['peak_incidence']:.0f}"},
                    {"label": "Neighbouring states", "value": str(len(sb.rows))}]
            facts = {"state": st, "population": int(prof["population"]), "districts": int(prof["n_districts"]),
                     "total_cases_end_2020": int(prof["total_cases"]), "deaths_2020": int(prof["deaths"]),
                     "peak_incidence": round(float(prof["peak_incidence"]), 1), "peak_date": str(prof["peak_date"])[:10],
                     "max_hotspot_districts_on_one_day": int(prof["max_hotspot_districts"]),
                     "neighbouring_states": [r["state"] for r in sb.rows]}
            text = (f"{st} (population {int(prof['population']):,}, {int(prof['n_districts'])} districts) recorded "
                    f"{int(prof['total_cases']):,} cases in 2020. Its 7-day incidence peaked at "
                    f"{prof['peak_incidence']:.0f} on {_d(prof['peak_date'])}.")
        fig = charts.lines(trend, f"7-day incidence per 100k – {name}, 2020", "7-day incidence per 100k",
                           config.HOTSPOT_THRESHOLD)
        return fig, facts, text, [], kpis

    # ==================================================================
    # GRAPH handlers
    # ==================================================================
    def _h_neighbours(self, p: QueryPlan):
        d = p.districts[0]
        nb = self._g(self.graph.neighbours(d["id"]), f"districts bordering {d['name']}")
        ids = [d["id"]] + [r["id"] for r in nb.rows]
        inner = self._g(self.graph.edges_among(ids), "borders among them")
        nodes = self._coords(ids)
        nodes["role"] = np.where(nodes["district_id"] == d["id"], "focus", "neighbour")
        fig = charts.network_map(nodes, [(r["a"], r["b"]) for r in inner.rows],
                                 f"Districts bordering {d['name']} ({len(nb.rows)})",
                                 roles={"focus": (SERIES[1], d["name"]), "neighbour": (FOCUS, "Bordering district")},
                                 height=560)
        other_states = sorted({r["state"] for r in nb.rows} - {self._state_of(d["id"])})
        facts = {"district": d["name"], "state": self._state_of(d["id"]), "neighbour_count": len(nb.rows),
                 "neighbours": [r["name"] for r in nb.rows], "neighbours_in_other_states": other_states,
                 "borders_from_geography_not_data": [r["name"] for r in nb.rows if r.get("source") == "geography"]}
        text = f"{d['name']} borders {len(nb.rows)} districts: {', '.join(r['name'] for r in nb.rows)}."
        if other_states:
            text += f" Some of them lie in another federal state ({', '.join(other_states)})."
        return fig, facts, text, nb.rows, []

    def _state_of(self, did: int) -> str:
        return self.districts.loc[self.districts["district_id"] == did, "state"].iloc[0]

    def _h_shortest_path(self, p: QueryPlan):
        a, b = p.districts[0], p.districts[1]
        res = self._g(self.graph.shortest_path(a["id"], b["id"]), "shortest chain of borders")
        path = res.rows
        if not path:
            return None, {}, f"No border path connects {a['name']} and {b['name']}.", [], []
        ids = [r["id"] for r in path]
        nodes = self._coords(ids)
        nodes["role"] = nodes["district_id"].map(lambda i: "start" if i == ids[0] else "end" if i == ids[-1] else "via")
        order = {i: k for k, i in enumerate(ids)}
        nodes = nodes.sort_values("district_id", key=lambda s: s.map(order))
        fig = charts.network_map(nodes, list(zip(ids, ids[1:])),
                                 f"Shortest border path: {a['name']} → {b['name']} ({len(ids) - 1} borders)",
                                 roles={"start": (SERIES[2], f"Start: {a['name']}"), "via": (FOCUS, "Passes through"),
                                        "end": (SERIES[1], f"End: {b['name']}")},
                                 background_edges=self._all_edge_pairs())
        states = list(dict.fromkeys(r["state"] for r in path))
        facts = {"from": a["name"], "to": b["name"], "borders_crossed": len(ids) - 1,
                 "path": [r["name"] for r in path], "states_on_path": states}
        text = (f"The shortest route from {a['name']} to {b['name']} crosses {len(ids) - 1} district borders, passing "
                f"through {', '.join(r['name'] for r in path[1:-1])} and {len(states)} federal states.")
        return fig, facts, text, [{"step": i, **r} for i, r in enumerate(path)], []

    def _h_most_connected(self, p: QueryPlan):
        st = p.states[0] if p.states else None
        res = self._g(self.graph.most_connected(p.n, st), "districts ranked by number of borders")
        df = pd.DataFrame(res.rows).rename(columns={"degree": "value"})
        where = f" in {st}" if st else ""
        fig = charts.bar_ranking(df, f"Districts{where} with the most neighbours", "Number of bordering districts")
        facts = {"ranking": [{"name": r["name"], "state": r["state"], "neighbours": r["degree"]} for r in res.rows],
                 "meaning": "Degree centrality: the number of districts a district shares a border with."}
        r0 = res.rows[0]
        text = f"{r0['name']} ({r0['state']}) has the most neighbours{where}: it borders {r0['degree']} districts."
        return fig, facts, text, res.rows, []

    def _h_state_borders(self, p: QueryPlan):
        st = p.states[0]
        res = self._g(self.graph.state_borders(st), f"states bordering {st}")
        if not res.rows:
            return None, {"state": st}, f"{st} does not border any other federal state in this graph.", [], []
        df = pd.DataFrame(res.rows).rename(columns={"state": "name", "shared_borders": "value"})
        fig = charts.bar_ranking(df, f"Federal states bordering {st}", "District-to-district borders shared")
        facts = {"state": st, "neighbouring_states": [{"state": r["state"], "shared_district_borders": r["shared_borders"]}
                                                      for r in res.rows]}
        text = (f"{st} borders {len(res.rows)} federal states: {', '.join(r['state'] for r in res.rows)}. "
                f"The longest shared frontier (by number of district borders) is with {res.rows[0]['state']}.")
        return fig, facts, text, res.rows, []

    # ==================================================================
    # HYBRID handlers  (graph finds WHO, SQL measures HOW MUCH)
    # ==================================================================
    def _metric_snapshot(self, p: QueryPlan, metric: str, ids: list) -> tuple[pd.DataFrame, str]:
        m = METRICS[metric]
        if p.start and p.end and m["period"] != "last":
            df = self._sql(self.sql.snapshot(metric, None, ids, p.start, p.end), f"{metric_label(metric)} for these districts")
        else:
            df = self._sql(self.sql.snapshot(metric, self._snapshot_date(p), ids), f"{metric_label(metric)} for these districts")
        return df, self._snapshot_desc(p, metric)

    def _h_k_hop(self, p: QueryPlan):
        d = p.districts[0]
        metric = p.metric if p.metric not in ("hotspot",) else "incidence_7d"
        lab = metric_label(metric)
        hop = self._g(self.graph.k_hop(d["id"], p.hops), f"districts within {p.hops} borders")      # GRAPH
        hops = pd.DataFrame(hop.rows)
        ids = [d["id"]] + hops["id"].tolist()
        vals, when = self._metric_snapshot(p, metric, ids)                                         # SQL
        df = vals.merge(hops[["id", "hops"]].rename(columns={"id": "district_id"}), on="district_id", how="left")   # FUSE
        df["hops"] = df["hops"].fillna(0).astype(int)
        inner = self._g(self.graph.edges_among(ids), "borders among them")
        shown = df if p.threshold is None else df[(df["value"] > p.threshold) | (df["hops"] == 0)]
        title = f"{lab} within {p.hops} borders of {d['name']}, {when}"
        if p.threshold is not None:
            title += f" (above {p.threshold:g} highlighted)"
            nodes = df.copy()
            nodes["role"] = np.where(nodes["hops"] == 0, "focus",
                                     np.where(nodes["value"] > p.threshold, "above", "below"))
            fig = charts.network_map(nodes, [(r["a"], r["b"]) for r in inner.rows], title,
                                     roles={"focus": (SERIES[1], d["name"]),
                                            "above": (HOTSPOT if metric == "incidence_7d" else FOCUS, f"Above {p.threshold:g}"),
                                            "below": (CONTEXT, f"At or below {p.threshold:g}")},
                                     value_label=lab)
        else:
            fig = charts.network_map(df, [(r["a"], r["b"]) for r in inner.rows], title, color_by="value", value_label=lab)
            focus = df[df["hops"] == 0]
            fig.add_trace(go.Scattergeo(lat=focus["lat"], lon=focus["lon"], text=focus["name"], mode="markers",
                                        name=d["name"], marker=dict(size=16, symbol="star", color=SERIES[1],
                                                                    line=dict(color="white", width=1.5)),
                                        hovertemplate="%{text} (start)<extra></extra>"))
        ring = df[df["hops"] > 0].groupby("hops")["value"].agg(["count", "mean"]).round(1)
        facts = {"centre": d["name"], "hops": p.hops, "metric": lab, "period": when,
                 "centre_value": round(float(df.loc[df["hops"] == 0, "value"].iloc[0]), 1),
                 "districts_found": int((df["hops"] > 0).sum()),
                 "by_distance": [{"borders_away": int(h), "districts": int(r["count"]), "average_value": float(r["mean"])}
                                 for h, r in ring.iterrows()],
                 "highest": [{"name": r["name"], "value": round(float(r["value"]), 1), "borders_away": int(r["hops"])}
                             for _, r in df[df["hops"] > 0].nlargest(3, "value").iterrows()]}
        if p.threshold is not None:
            facts["threshold"] = p.threshold
            facts["districts_above_threshold"] = int(((df["value"] > p.threshold) & (df["hops"] > 0)).sum())
            facts["names_above_threshold"] = shown[shown["hops"] > 0].sort_values("value", ascending=False)["name"].tolist()[:15]
        text = (f"{facts['districts_found']} districts lie within {p.hops} borders of {d['name']} "
                f"(graph traversal); their {lab.lower()} {when} was then looked up in SQL.")
        if p.threshold is not None:
            text += f" {facts['districts_above_threshold']} of them are above {p.threshold:g}."
        text += " " + "; ".join(f"{b['borders_away']} border(s) away: average {fmt(b['average_value'])}" for b in facts["by_distance"]) + "."
        table = self._records(shown.sort_values(["hops", "value"], ascending=[True, False])
                              [["district_id", "name", "state", "hops", "value"]])
        return fig, facts, text, table, []

    def _h_neighbour_metrics(self, p: QueryPlan):
        d = p.districts[0]
        metric = p.metric if p.metric != "hotspot" else "incidence_7d"
        lab = metric_label(metric)
        nb = self._g(self.graph.neighbours(d["id"]), f"districts bordering {d['name']}")                # GRAPH
        ids = [d["id"]] + [r["id"] for r in nb.rows]
        df, when = self._metric_snapshot(p, metric, ids)                                                  # SQL
        hot = metric == "incidence_7d"
        sel = f"{d['name']} (selected)"
        plot_df = df.assign(name=df["name"].where(df["district_id"] != d["id"], sel))
        fig = charts.bar_ranking(plot_df, f"{lab}: {d['name']} and its {len(nb.rows)} neighbours, {when}", lab,
                                 "hotspot" if hot else "highlight", focus_names=[sel])
        focus_v = float(df.loc[df["district_id"] == d["id"], "value"].iloc[0])
        others = df[df["district_id"] != d["id"]]
        rank = int((df["value"] > focus_v).sum()) + 1
        facts = {"district": d["name"], "metric": lab, "period": when, "district_value": round(focus_v, 1),
                 "neighbour_average": round(float(others["value"].mean()), 1),
                 "neighbour_min": round(float(others["value"].min()), 1), "neighbour_max": round(float(others["value"].max()), 1),
                 "rank_of_district_in_group": f"{rank} of {len(df)}",
                 "neighbours": [{"name": r["name"], "value": round(float(r["value"]), 1)} for _, r in others.iterrows()]}
        if hot:
            facts["neighbour_hotspots"] = int((others["value"] >= config.HOTSPOT_THRESHOLD).sum())
        rel = "above" if focus_v > facts["neighbour_average"] else "below"
        text = (f"{d['name']} had {fmt(focus_v)} ({lab.lower()}, {when}), {rel} the average of its "
                f"{len(others)} neighbours ({fmt(facts['neighbour_average'])}); it ranks {rank} of {len(df)} in this group.")
        if hot:
            text += f" {facts['neighbour_hotspots']} of the neighbours are hotspots."
        return fig, facts, text, self._records(df.sort_values("value", ascending=False)), []

    def _hotspot_set(self, p: QueryPlan) -> tuple[list, str]:
        day = self._snapshot_date(p)
        ids = self._sql(self.sql.hotspot_ids(day), f"hotspot districts on {day}")["district_id"].tolist()   # SQL
        return ids, day

    def _h_hotspot_clusters(self, p: QueryPlan):
        ids, day = self._hotspot_set(p)
        if not ids:
            return None, {"date": day, "hotspots": 0}, f"There were no hotspot districts on {_d(day)}.", [], []
        cl = self._g(self.graph.clusters(ids), "groups of bordering hotspots")                              # GRAPH
        nodes = self._coords(ids)
        role_of = {}
        roles = {}
        big = [c for c in cl.rows if c["size"] > 1]
        for i, c in enumerate(big[:7]):
            roles[f"c{i}"] = (SERIES[i], f"Cluster {i + 1} ({c['size']} districts)")
            role_of.update({x: f"c{i}" for x in c["ids"]})
        roles["other"] = (CONTEXT, "Smaller clusters / single hotspots")
        nodes["role"] = nodes["district_id"].map(lambda x: role_of.get(x, "other"))
        fig = charts.network_map(nodes, [(e["a"], e["b"]) for e in cl.extra["edges"]],
                                 f"Clusters of bordering hotspot districts, {_d(day)}", roles=roles,
                                 background_edges=self._all_edge_pairs())
        st = self.districts.set_index("district_id")["state"]
        facts = {"date": day, "hotspots": len(ids), "clusters": len(cl.rows),
                 "clusters_with_2_or_more": len(big), "single_isolated_hotspots": sum(1 for c in cl.rows if c["size"] == 1),
                 "largest_clusters": [{"size": c["size"], "states": sorted(st[c["ids"]].unique().tolist())} for c in cl.rows[:3]]}
        largest = cl.rows[0]
        text = (f"On {_d(day)} there were {len(ids)} hotspot districts. Following shared borders, they form "
                f"{len(cl.rows)} connected groups; the largest joins {largest['size']} districts across "
                f"{len(facts['largest_clusters'][0]['states'])} federal states.")
        table = [{"cluster": c["cluster"], "size": c["size"],
                  "districts": ", ".join(self.districts.set_index('district_id').loc[c['ids'][:12], 'name'])
                  + (" …" if c["size"] > 12 else "")} for c in cl.rows[:20]]
        return fig, facts, text, table, []

    def _surround_common(self, p: QueryPlan, kind: str):
        ids, day = self._hotspot_set(p)
        if not ids:
            return None, {"date": day, "hotspots": 0}, f"There were no hotspot districts on {_d(day)}.", [], []
        res = self._g(self.graph.surrounded(ids) if kind == "surrounded" else self.graph.isolated(ids),
                      "hotspots whose neighbours are all hotspots" if kind == "surrounded" else "hotspots with no hotspot neighbour")
        sel = {r["id"] for r in res.rows}
        nodes = self.districts.copy()
        nodes["role"] = np.where(nodes["district_id"].isin(sel), "sel",
                                 np.where(nodes["district_id"].isin(ids), "hot", "none"))
        label = "Hotspot surrounded only by hotspots" if kind == "surrounded" else "Isolated hotspot (no hotspot neighbour)"
        title = ("Hotspots surrounded only by hotspots" if kind == "surrounded"
                 else "Isolated hotspots (no hotspot neighbour)")
        fig = charts.network_map(nodes, [], f"{title}, {_d(day)}",
                                 roles={"sel": (HOTSPOT, label), "hot": ("#ec835a", "Other hotspot"),
                                        "none": ("rgba(137,135,129,0.35)", "Not a hotspot")},
                                 background_edges=self._all_edge_pairs())
        facts = {"date": day, "hotspots": len(ids), "count": len(res.rows),
                 "share_of_hotspots_pct": round(100 * len(res.rows) / len(ids), 1),
                 "examples": [r["name"] for r in res.rows[:10]]}
        if kind == "surrounded":
            text = (f"On {_d(day)}, {len(res.rows)} of the {len(ids)} hotspot districts ({facts['share_of_hotspots_pct']}%) "
                    f"were completely surrounded by other hotspots, a sign of regional rather than local outbreaks.")
        else:
            text = (f"On {_d(day)}, {len(res.rows)} of the {len(ids)} hotspot districts had no hotspot neighbour at all, "
                    f"which points to local outbreaks.")
        return fig, facts, text, res.rows, []

    def _h_surrounded_hotspots(self, p: QueryPlan):
        return self._surround_common(p, "surrounded")

    def _h_isolated_hotspots(self, p: QueryPlan):
        return self._surround_common(p, "isolated")


_orch: Orchestrator | None = None


def get_orchestrator() -> Orchestrator:
    global _orch
    if _orch is None:
        _orch = Orchestrator()
    return _orch


def system_status() -> dict:
    from backend.llm import get_llm
    return {"graph": graph_status(), "llm": get_llm().status(), "duckdb": str(config.DUCKDB_PATH),
            "intents": {k: v[0] for k, v in INTENTS.items()}}
