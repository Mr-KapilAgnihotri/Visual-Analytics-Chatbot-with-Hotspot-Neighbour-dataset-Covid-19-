"""
Builds the CONTENT of the knowledge graph (nodes, relationships, properties)
from DuckDB + the recovered borders. Both graph backends (Neo4j and the
NetworkX fallback) are built from this one payload, so they always hold the
same knowledge.

Knowledge-graph schema
----------------------
  (:State {name, population, n_districts})
  (:District {id, name, state, population, lat, lon,
              peak_incidence, peak_date, days_as_hotspot, first_hotspot_date,
              total_cases, total_deaths})
  (:Month {ym})                                   e.g. "2020-11"
  (:District)-[:IN_STATE]->(:State)
  (:District)-[:BORDERS {source}]-(:District)     source = "data" | "geography"
  (:State)-[:BORDERS {shared_borders}]-(:State)   derived from district borders
  (:District)-[:HOTSPOT_IN {days, peak}]->(:Month)

Note what is NOT in the graph: the 150k daily rows. The graph holds entities,
relations and summaries; the time series lives in SQL, where it belongs.
"""
import json
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.config import ADJ_JSON, DUCKDB_PATH  # noqa: E402


def build_payload(db_path: Path = DUCKDB_PATH) -> dict:
    con = duckdb.connect(str(db_path), read_only=True)
    districts = con.execute("""
        WITH peak AS (
            SELECT district_id, MAX(incidence_7d) AS peak_incidence,
                   ARG_MAX(date, incidence_7d)  AS peak_date,
                   SUM(CAST(hotspot AS INTEGER)) AS days_as_hotspot,
                   MIN(CASE WHEN hotspot THEN date END) AS first_hotspot_date,
                   MAX(total_cases) AS total_cases, SUM(deaths) AS total_deaths
            FROM fact_daily GROUP BY district_id)
        SELECT d.district_id AS id, d.district AS name, s.state, d.population, d.lat, d.lon,
               p.peak_incidence, CAST(p.peak_date AS VARCHAR) AS peak_date, p.days_as_hotspot,
               CAST(p.first_hotspot_date AS VARCHAR) AS first_hotspot_date,
               p.total_cases, p.total_deaths
        FROM dim_district d JOIN dim_state s USING (state_id) JOIN peak p USING (district_id)
        ORDER BY id
    """).df()
    states = con.execute("""
        SELECT s.state AS name, SUM(d.population) AS population, COUNT(*) AS n_districts
        FROM dim_district d JOIN dim_state s USING (state_id) GROUP BY 1 ORDER BY 1
    """).df()
    hotspot_months = con.execute("""
        SELECT district_id AS id, STRFTIME(date, '%Y-%m') AS ym,
               COUNT(*) AS days, MAX(incidence_7d) AS peak
        FROM fact_daily WHERE hotspot GROUP BY 1, 2 ORDER BY 1, 2
    """).df()
    con.close()

    edges = json.loads(Path(ADJ_JSON).read_text())["edges"]
    borders = [{"a": a, "b": b, "source": s} for a, b, s in edges]

    state_of = dict(zip(districts["id"], districts["state"]))
    pair_count: dict = {}
    for e in borders:
        sa, sb = state_of[e["a"]], state_of[e["b"]]
        if sa != sb:
            key = tuple(sorted((sa, sb)))
            pair_count[key] = pair_count.get(key, 0) + 1
    state_borders = [{"a": a, "b": b, "shared_borders": n} for (a, b), n in sorted(pair_count.items())]

    def records(frame):
        return json.loads(frame.to_json(orient="records"))   # plain python types, NaN -> None

    return {
        "states": records(states),
        "districts": records(districts),
        "borders": borders,
        "state_borders": state_borders,
        "hotspot_months": records(hotspot_months),
        "months": [f"2020-{m:02d}" for m in range(1, 13)],
    }


if __name__ == "__main__":
    p = build_payload()
    print({k: len(v) for k, v in p.items()})
