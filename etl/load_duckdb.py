"""
Step 3 of the ETL pipeline: load the relational side (DuckDB).

Star schema
-----------
    dim_state      (state_id PK, state)
    dim_district   (district_id PK, district, state_id FK, population, lat, lon)
    fact_daily     (district_id FK, date, ...all daily measures...)   PK(district_id, date)
    district_border(a, b, source)   -- both directions; used ONLY by the benchmark to
                                       show how multi-hop queries look in SQL
    v_daily        -- convenience view: fact_daily joined with its dimensions

Why a star schema? The facts (numbers per district per day) are separated from
the descriptions (names, states, coordinates), so nothing descriptive is repeated
150,000 times, and every analytical query is "filter facts, join dims, group by".

The script is IDEMPOTENT: it drops and recreates everything, so running it twice
gives exactly the same database.

Run:  python -m etl.load_duckdb
"""
import json
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.config import ADJ_JSON, CLEAN_PARQUET, DUCKDB_PATH  # noqa: E402


def load(db_path: Path = DUCKDB_PATH) -> None:
    t0 = time.perf_counter()
    df = pd.read_parquet(CLEAN_PARQUET)
    edges = json.loads(Path(ADJ_JSON).read_text())["edges"]

    con = duckdb.connect(str(db_path))
    con.execute("DROP VIEW IF EXISTS v_daily")
    for t in ["fact_daily", "district_border", "dim_district", "dim_state"]:
        con.execute(f"DROP TABLE IF EXISTS {t}")

    states = sorted(df["state"].unique())
    state_df = pd.DataFrame({"state_id": range(1, len(states) + 1), "state": states})
    con.register("state_df", state_df)
    con.execute("""
        CREATE TABLE dim_state (state_id INTEGER PRIMARY KEY, state VARCHAR NOT NULL UNIQUE);
        INSERT INTO dim_state SELECT * FROM state_df;
    """)

    dist = (df.drop_duplicates("district_id")
              .merge(state_df, on="state")[["district_id", "district", "state_id", "population", "lat", "lon"]])
    con.register("dist_df", dist)
    con.execute("""
        CREATE TABLE dim_district (
            district_id INTEGER PRIMARY KEY, district VARCHAR NOT NULL,
            state_id INTEGER REFERENCES dim_state(state_id),
            population INTEGER, lat DOUBLE, lon DOUBLE);
        INSERT INTO dim_district SELECT * FROM dist_df;
    """)

    fact_cols = ["district_id", "date", "total_cases", "new_cases", "cases_7d", "incidence_7d",
                 "hotspot", "deaths", "mobility_inward", "mobility_value", "mobility_pct",
                 "mobility_change", "nb_total_cases", "nb_new_cases", "nb_cases_7d",
                 "nb_incidence_7d", "nb_deaths"]
    con.register("fact_df", df[fact_cols])
    con.execute("""
        CREATE TABLE fact_daily (
            district_id INTEGER REFERENCES dim_district(district_id), date DATE,
            total_cases INTEGER, new_cases INTEGER, cases_7d INTEGER, incidence_7d DOUBLE,
            hotspot BOOLEAN, deaths INTEGER, mobility_inward DOUBLE, mobility_value DOUBLE,
            mobility_pct DOUBLE, mobility_change DOUBLE, nb_total_cases INTEGER,
            nb_new_cases INTEGER, nb_cases_7d INTEGER, nb_incidence_7d DOUBLE, nb_deaths INTEGER,
            PRIMARY KEY (district_id, date));
        INSERT INTO fact_daily SELECT * FROM fact_df ORDER BY date, district_id;
    """)

    edge_df = pd.DataFrame(edges, columns=["a", "b", "source"])
    both = pd.concat([edge_df, edge_df.rename(columns={"a": "b", "b": "a"})])
    con.register("edge_df", both[["a", "b", "source"]])
    con.execute("""
        CREATE TABLE district_border (a INTEGER, b INTEGER, source VARCHAR, PRIMARY KEY (a, b));
        INSERT INTO district_border SELECT * FROM edge_df;
    """)

    con.execute("""
        CREATE VIEW v_daily AS
        SELECT f.*, d.district, d.population, d.lat, d.lon, s.state
        FROM fact_daily f
        JOIN dim_district d USING (district_id)
        JOIN dim_state s USING (state_id);
    """)
    con.execute("CREATE INDEX idx_fact_date ON fact_daily(date)")

    n = con.execute("SELECT COUNT(*) FROM fact_daily").fetchone()[0]
    nd = con.execute("SELECT COUNT(*) FROM dim_district").fetchone()[0]
    ne = con.execute("SELECT COUNT(*) FROM district_border").fetchone()[0] // 2
    con.close()
    print(f"DuckDB loaded in {time.perf_counter() - t0:.1f}s: {n:,} facts, {nd} districts, "
          f"{len(states)} states, {ne} borders -> {db_path}")


if __name__ == "__main__":
    load()
