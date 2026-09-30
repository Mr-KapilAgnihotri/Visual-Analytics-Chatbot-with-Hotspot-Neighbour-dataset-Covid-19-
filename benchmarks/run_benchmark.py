"""
BENCHMARK: which engine is faster for which kind of question?

Two families of workload, each run on every available engine:

  A. GRAPH-SHAPED (follow relationships)
       A1  k-hop neighbourhood, k = 1..8     Cypher shortestPath | SQL recursive CTE | NetworkX BFS
       A2  hotspots surrounded by hotspots   Cypher list predicate | SQL NOT EXISTS anti-join | NetworkX
  B. AGGREGATION-SHAPED (scan & summarise the 150k-row time series)
       B1  top 10 districts by incidence on one day
       B2  monthly new cases per federal state (GROUP BY)
     For B, Neo4j needs the daily rows as nodes. Pass --with-observations to load
     149,592 temporary (:Observation) nodes; they are deleted afterwards.

Each measurement is the MEDIAN of --repeats runs after one warm-up run.
Results: benchmarks/results.csv and benchmarks/results.html (interactive chart).

Run:  python -m benchmarks.run_benchmark [--with-observations] [--repeats 20]
"""
from __future__ import annotations

import argparse
import logging
import statistics
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend import config  # noqa: E402
from backend.charts import SERIES, _layout  # noqa: E402
from backend.graph_engine import Neo4jGraph, NetworkXGraph  # noqa: E402
from backend.sql_engine import SqlEngine  # noqa: E402

logging.disable(logging.WARNING)
OUT = Path(__file__).resolve().parent
CENTRE = 9162            # München (city)
DAY = "2020-12-01"


def median_ms(fn, repeats: int) -> float:
    fn()                                              # warm-up (caches, query plans)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


SQL_SURROUNDED = """
WITH hot AS (SELECT district_id FROM fact_daily WHERE date = ? AND hotspot)
SELECT h.district_id FROM hot h
WHERE EXISTS (SELECT 1 FROM district_border b WHERE b.a = h.district_id)
  AND NOT EXISTS (SELECT 1 FROM district_border b
                  WHERE b.a = h.district_id AND b.b NOT IN (SELECT district_id FROM hot))"""

SQL_TOP10 = """
SELECT d.district, f.incidence_7d FROM fact_daily f JOIN dim_district d USING (district_id)
WHERE f.date = ? ORDER BY f.incidence_7d DESC LIMIT 10"""
SQL_MONTHLY = """
SELECT s.state, DATE_TRUNC('month', f.date) AS month, SUM(f.new_cases) AS new_cases
FROM fact_daily f JOIN dim_district d USING (district_id) JOIN dim_state s USING (state_id)
GROUP BY 1, 2"""

CY_SURROUNDED = """
MATCH (d:District) WHERE d.id IN $ids
WITH d, [(d)-[:BORDERS]-(n:District) | n.id] AS nbrs
WHERE size(nbrs) > 0 AND all(x IN nbrs WHERE x IN $ids)
RETURN d.id"""
CY_TOP10 = """
MATCH (o:Observation {date: date($day)})
RETURN o.district_id, o.incidence_7d ORDER BY o.incidence_7d DESC LIMIT 10"""
CY_MONTHLY = """
MATCH (o:Observation)
RETURN o.state AS state, o.month AS month, sum(o.new_cases) AS new_cases"""


def load_observations(neo: Neo4jGraph, sql: SqlEngine) -> None:
    df = sql.run("""SELECT f.district_id, CAST(f.date AS VARCHAR) AS date, f.new_cases, f.incidence_7d,
                           s.state, STRFTIME(f.date, '%Y-%m') AS month
                    FROM fact_daily f JOIN dim_district d USING (district_id) JOIN dim_state s USING (state_id)""").df
    neo._query("CREATE INDEX obs_date IF NOT EXISTS FOR (o:Observation) ON (o.date)")
    rows = df.to_dict("records")
    for i in range(0, len(rows), 10000):
        neo._query("UNWIND $rows AS r CREATE (:Observation {district_id: r.district_id, date: date(r.date), "
                   "new_cases: r.new_cases, incidence_7d: r.incidence_7d, state: r.state, month: r.month})",
                   rows=rows[i:i + 10000])
    print(f"  loaded {len(rows):,} temporary :Observation nodes into Neo4j")


def drop_observations(neo: Neo4jGraph) -> None:
    while neo._query("MATCH (o:Observation) WITH o LIMIT 10000 DETACH DELETE o RETURN count(*) AS n")[0]["n"]:
        pass
    print("  removed temporary :Observation nodes")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--max-k", type=int, default=8)
    ap.add_argument("--with-observations", action="store_true")
    args = ap.parse_args()

    sql = SqlEngine()
    nx_g = NetworkXGraph()
    try:
        neo = Neo4jGraph()
        print(f"Neo4j connected at {config.NEO4J_URI}")
    except Exception as exc:  # noqa: BLE001
        neo = None
        print(f"Neo4j not available ({exc.__class__.__name__}); benchmarking DuckDB and NetworkX only")

    results = []

    def record(workload, family, param, engine, ms, rows=None):
        results.append({"workload": workload, "family": family, "param": param, "engine": engine,
                        "median_ms": round(ms, 3), "rows": rows})
        print(f"  {workload:<28} {str(param):>4}  {engine:<22} {ms:9.3f} ms" + (f"  ({rows} rows)" if rows is not None else ""))

    print("\nA1  k-hop neighbourhood of München")
    for k in range(1, args.max_k + 1):
        rows = len(sql.k_hop_recursive(CENTRE, k).df)
        record("A1 k-hop", "graph-shaped", k, "DuckDB (recursive CTE)",
               median_ms(lambda: sql.k_hop_recursive(CENTRE, k), args.repeats), rows)
        if neo:
            record("A1 k-hop", "graph-shaped", k, "Neo4j (Cypher)",
                   median_ms(lambda: neo.k_hop(CENTRE, k), args.repeats), len(neo.k_hop(CENTRE, k).rows))
        record("A1 k-hop", "graph-shaped", k, "NetworkX (in-memory)",
               median_ms(lambda: nx_g.k_hop(CENTRE, k), args.repeats), len(nx_g.k_hop(CENTRE, k).rows))

    print(f"\nA2  hotspots surrounded by hotspots on {DAY}")
    ids = sql.hotspot_ids(DAY).df["district_id"].tolist()
    record("A2 surrounded hotspots", "graph-shaped", "-", "DuckDB (anti-join)",
           median_ms(lambda: sql.run(SQL_SURROUNDED, [DAY]), args.repeats), len(sql.run(SQL_SURROUNDED, [DAY]).df))
    if neo:
        record("A2 surrounded hotspots", "graph-shaped", "-", "Neo4j (Cypher)",
               median_ms(lambda: neo._query(CY_SURROUNDED, ids=ids), args.repeats), len(neo._query(CY_SURROUNDED, ids=ids)))
    record("A2 surrounded hotspots", "graph-shaped", "-", "NetworkX (in-memory)",
           median_ms(lambda: nx_g.surrounded(ids), args.repeats), len(nx_g.surrounded(ids).rows))

    print("\nB1/B2  aggregations over the daily time series")
    record("B1 top-10 on one day", "aggregation", "-", "DuckDB (SQL)",
           median_ms(lambda: sql.run(SQL_TOP10, [DAY]), args.repeats))
    record("B2 monthly cases by state", "aggregation", "-", "DuckDB (SQL)",
           median_ms(lambda: sql.run(SQL_MONTHLY), args.repeats))
    if neo and args.with_observations:
        load_observations(neo, sql)
        try:
            record("B1 top-10 on one day", "aggregation", "-", "Neo4j (Cypher)",
                   median_ms(lambda: neo._query(CY_TOP10, day=DAY), args.repeats))
            record("B2 monthly cases by state", "aggregation", "-", "Neo4j (Cypher)",
                   median_ms(lambda: neo._query(CY_MONTHLY), max(3, args.repeats // 4)))
        finally:
            drop_observations(neo)
    elif neo:
        print("  (add --with-observations to also time Neo4j on the aggregations)")

    df = pd.DataFrame(results)
    df.to_csv(OUT / "results.csv", index=False)
    make_report(df)
    print(f"\nSaved {OUT / 'results.csv'} and {OUT / 'results.html'}")


def make_report(df: pd.DataFrame) -> None:
    engines = list(dict.fromkeys(df["engine"].str.split(" ").str[0]))
    color = {e: SERIES[i] for i, e in enumerate(engines)}
    k = df[df["workload"] == "A1 k-hop"]
    fig1 = go.Figure()
    for eng, g in k.groupby("engine", sort=False):
        fig1.add_scatter(x=g["param"], y=g["median_ms"], mode="lines+markers", name=eng,
                         line=dict(color=color[eng.split(" ")[0]], width=2), marker=dict(size=8))
    _layout(fig1, "A1 · k-hop neighbourhood: time vs number of hops (median, log scale)",
            "hops (k)", "milliseconds")
    fig1.update_yaxes(type="log")
    rest = df[df["workload"] != "A1 k-hop"]
    fig2 = go.Figure()
    for eng, g in rest.groupby("engine", sort=False):
        fig2.add_bar(x=g["workload"], y=g["median_ms"], name=eng, marker=dict(color=color[eng.split(" ")[0]], cornerradius=4))
    _layout(fig2, "Pattern matching (A2) and aggregations (B1, B2): median time", "", "milliseconds")
    fig2.update_layout(barmode="group")
    html = ["<html><head><meta charset='utf-8'><title>Engine benchmark</title></head>"
            "<body style='font-family:system-ui;max-width:1000px;margin:auto;padding:16px'>",
            "<h2>Knowledge graph vs relational database: benchmark</h2>",
            "<p>Median of repeated runs after a warm-up. NetworkX runs inside the Python process (no query "
            "parsing, no network), so it shows the cost of the traversal itself; Neo4j and DuckDB include their "
            "query-engine overhead.</p>",
            fig1.to_html(full_html=False, include_plotlyjs="cdn"), fig2.to_html(full_html=False, include_plotlyjs=False),
            df.to_html(index=False), "</body></html>"]
    (OUT / "results.html").write_text("\n".join(html), encoding="utf-8")


if __name__ == "__main__":
    main()
