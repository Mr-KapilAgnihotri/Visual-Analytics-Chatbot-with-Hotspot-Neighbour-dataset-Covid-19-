"""
The RELATIONAL engine (DuckDB).

Every question type is a parameterised SQL TEMPLATE. User values (dates, names,
numbers) are passed as bound parameters (`?`), never pasted into the SQL text,
so the chatbot cannot be tricked into running arbitrary SQL (SQL injection).
The only things formatted into the text are metric column names and SQL
expressions, and those come from the whitelist in catalog.METRICS.

Why these questions belong to SQL: they scan and aggregate the 150k-row time
series (SUM / AVG / GROUP BY / ORDER BY ... LIMIT). DuckDB is a columnar,
vectorised OLAP engine: it reads only the columns a query needs and processes
them in batches of thousands of values at a time.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import duckdb
import pandas as pd

from backend import config
from backend.catalog import METRICS


@dataclass
class SqlResult:
    df: pd.DataFrame
    query: str
    ms: float
    language: str = "SQL"


def _m(metric: str) -> dict:
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}")
    return METRICS[metric]


class SqlEngine:
    def __init__(self, db_path=config.DUCKDB_PATH):
        self.con = duckdb.connect(str(db_path), read_only=True)
        self._lock = threading.Lock()

    # ---------------------------------------------------------------------
    def run(self, sql: str, params: list | None = None) -> SqlResult:
        t0 = time.perf_counter()
        with self._lock:                       # one DuckDB cursor per call, thread-safe
            cur = self.con.cursor()
            df = cur.execute(sql, params or []).df()
            cur.close()
        return SqlResult(df, self._pretty(sql, params), (time.perf_counter() - t0) * 1000)

    @staticmethod
    def _pretty(sql: str, params) -> str:
        """SQL text with parameter values shown as comments, for display only."""
        text = "\n".join(line.rstrip() for line in sql.strip().splitlines())
        if params:
            text += "\n-- parameters: " + ", ".join(repr(p) for p in params)
        return text

    # ---------------------------------------------------------------------
    # helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _filters(districts=None, states=None, alias="") -> tuple[str, list]:
        a = f"{alias}." if alias else ""
        parts, params = [], []
        if districts:
            parts.append(f"{a}district_id IN ({','.join('?' * len(districts))})")
            params += list(districts)
        if states:
            parts.append(f"{a}state IN ({','.join('?' * len(states))})")
            params += list(states)
        if not parts:
            return "", []
        return " AND (" + " OR ".join(parts) + ")", params

    def _period_value(self, metric: str) -> str:
        """Expression that turns many days of ONE district into one number."""
        m = _m(metric)
        col = m["col"]
        if metric == "hotspot":
            return "SUM(CAST(hotspot AS INTEGER))"
        return {"sum": f"SUM({col})", "avg": f"AVG({col})", "last": f"ARG_MAX({col}, date)",
                "max": f"MAX({col})"}[m["period"]]

    # ---------------------------------------------------------------------
    # templates
    # ---------------------------------------------------------------------
    def top_n(self, metric, level="district", date=None, start=None, end=None, n=10,
              order="desc", states=None, peak=False) -> SqlResult:
        m = _m(metric)
        order_sql = "ASC" if order == "asc" else "DESC"
        n = max(1, min(int(n), 50))
        sfilter, sparams = self._filters(states=states)
        if level == "state":
            if start and end and m["period"] in ("sum", "avg"):
                sql = f"""
SELECT state AS name, {m['group_sql']} AS value
FROM v_daily
WHERE date BETWEEN ? AND ?{sfilter}
GROUP BY state ORDER BY value {order_sql} LIMIT {n}"""
                return self.run(sql, [start, end] + sparams)
            sql = f"""
SELECT state AS name, {m['group_sql']} AS value
FROM v_daily
WHERE date = ?{sfilter}
GROUP BY state ORDER BY value {order_sql} LIMIT {n}"""
            return self.run(sql, [end or date or config.LAST_DATE] + sparams)

        # district level
        if start and end:
            expr = f"MAX({m['col']})" if peak else self._period_value(metric)
            sql = f"""
SELECT district_id, district AS name, state, {expr} AS value
FROM v_daily
WHERE date BETWEEN ? AND ?{sfilter}
GROUP BY district_id, district, state
ORDER BY value {order_sql}, name LIMIT {n}"""
            return self.run(sql, [start, end] + sparams)
        sql = f"""
SELECT district_id, district AS name, state, {m['col']} AS value
FROM v_daily
WHERE date = ?{sfilter}
ORDER BY value {order_sql}, name LIMIT {n}"""
        return self.run(sql, [date or config.LAST_DATE] + sparams)

    def trend(self, metric, districts=None, states=None, start=None, end=None,
              granularity="day") -> SqlResult:
        """One time series per requested district / state, or for all of Germany."""
        m = _m(metric)
        start, end = start or config.FIRST_DATE, end or config.LAST_DATE
        gran = granularity if granularity in ("day", "week", "month") else "day"
        bucket_agg = "SUM" if m["period"] == "sum" else "AVG"
        blocks, params = [], []
        for did in districts or []:
            blocks.append(f"""
    SELECT date, district AS series, {m['group_sql']} AS value
    FROM v_daily WHERE district_id = ? AND date BETWEEN ? AND ?
    GROUP BY date, district""")
            params += [did, start, end]
        for st in states or []:
            blocks.append(f"""
    SELECT date, state AS series, {m['group_sql']} AS value
    FROM v_daily WHERE state = ? AND date BETWEEN ? AND ?
    GROUP BY date, state""")
            params += [st, start, end]
        if not blocks:
            blocks.append(f"""
    SELECT date, 'Germany' AS series, {m['group_sql']} AS value
    FROM v_daily WHERE date BETWEEN ? AND ?
    GROUP BY date""")
            params += [start, end]
        sql = f"""
WITH daily AS ({' UNION ALL '.join(blocks)}
)
SELECT DATE_TRUNC('{gran}', date) AS period, series, {bucket_agg}(value) AS value
FROM daily GROUP BY period, series ORDER BY series, period"""
        return self.run(sql, params)

    def share_by_state(self, metric, date=None, start=None, end=None) -> SqlResult:
        m = _m(metric)
        if m["period"] == "sum":
            s, e = start or config.FIRST_DATE, end or date or config.LAST_DATE
            sql = f"""
SELECT state AS name, {m['group_sql']} AS value
FROM v_daily WHERE date BETWEEN ? AND ?
GROUP BY state ORDER BY value DESC"""
            return self.run(sql, [s, e])
        sql = f"""
SELECT state AS name, {m['group_sql']} AS value
FROM v_daily WHERE date = ?
GROUP BY state ORDER BY value DESC"""
        return self.run(sql, [end or date or config.LAST_DATE])

    def compare(self, metric, districts=None, states=None, date=None, start=None, end=None) -> SqlResult:
        m = _m(metric)
        blocks, params = [], []
        use_period = bool(start and end) and m["period"] in ("sum", "avg")
        where = "date BETWEEN ? AND ?" if use_period else "date = ?"
        dparams = [start, end] if use_period else [end or date or config.LAST_DATE]
        expr = m["group_sql"]
        for did in districts or []:
            blocks.append(f"SELECT district AS name, 'district' AS kind, {expr} AS value "
                          f"FROM v_daily WHERE district_id = ? AND {where} GROUP BY district")
            params += [did] + dparams
        for st in states or []:
            blocks.append(f"SELECT state AS name, 'state' AS kind, {expr} AS value "
                          f"FROM v_daily WHERE state = ? AND {where} GROUP BY state")
            params += [st] + dparams
        sql = "\nUNION ALL\n".join(blocks) + "\nORDER BY value DESC"
        return self.run(sql, params)

    def correlation(self, x_metric, y_metric, date=None, start=None, end=None,
                    level="district") -> SqlResult:
        """One point per district (or per state). Over a period: flows summed, levels averaged.
        State level first builds each state's DAILY value (population-weighted where needed),
        then averages/sums those over the period."""
        mx, my = _m(x_metric), _m(y_metric)
        if level == "state":
            s, e = (start, end) if start and end else (date or config.LAST_DATE,) * 2
            def agg(m):
                return "SUM" if m["period"] == "sum" else "AVG"
            sql = f"""
WITH daily AS (
    SELECT state, date, {mx['group_sql']} AS x, {my['group_sql']} AS y
    FROM v_daily WHERE date BETWEEN ? AND ?
    GROUP BY state, date)
SELECT state AS name, state, {agg(mx)}(x) AS x, {agg(my)}(y) AS y
FROM daily GROUP BY state"""
            return self.run(sql, [s, e])
        if start and end:
            def agg(m):
                return f"SUM({m['col']})" if m["period"] == "sum" else f"AVG({m['col']})"
            sql = f"""
SELECT district_id, district AS name, state,
       {agg(mx)} AS x, {agg(my)} AS y
FROM v_daily WHERE date BETWEEN ? AND ?
GROUP BY district_id, district, state"""
            return self.run(sql, [start, end])
        sql = f"""
SELECT district_id, district AS name, state, {mx['col']} AS x, {my['col']} AS y
FROM v_daily WHERE date = ?"""
        return self.run(sql, [date or config.LAST_DATE])

    def hotspot_trend(self, states=None, start=None, end=None) -> SqlResult:
        start, end = start or config.FIRST_DATE, end or config.LAST_DATE
        if states:
            ph = ",".join("?" * len(states))
            sql = f"""
SELECT date AS period, state AS series, SUM(CAST(hotspot AS INTEGER)) AS value
FROM v_daily WHERE date BETWEEN ? AND ? AND state IN ({ph})
GROUP BY date, state ORDER BY series, period"""
            return self.run(sql, [start, end] + list(states))
        sql = """
SELECT date AS period, 'Germany' AS series, SUM(CAST(hotspot AS INTEGER)) AS value
FROM v_daily WHERE date BETWEEN ? AND ?
GROUP BY date ORDER BY period"""
        return self.run(sql, [start, end])

    def snapshot(self, metric, date=None, ids=None, start=None, end=None) -> SqlResult:
        """Metric value for every district (or the given ids) with coordinates - for maps."""
        m = _m(metric)
        idf = ""
        params: list = []
        if ids is not None:
            ids = list(map(int, ids)) or [-1]
            idf = f" AND district_id IN ({','.join('?' * len(ids))})"
        if start and end:
            sql = f"""
SELECT district_id, district AS name, state, lat, lon, population,
       {self._period_value(metric)} AS value
FROM v_daily WHERE date BETWEEN ? AND ?{idf}
GROUP BY district_id, district, state, lat, lon, population"""
            params = [start, end] + (ids or [])
        else:
            sql = f"""
SELECT district_id, district AS name, state, lat, lon, population, {m['col']} AS value
FROM v_daily WHERE date = ?{idf}"""
            params = [date or config.LAST_DATE] + (ids or [])
        return self.run(sql, params)

    def hotspot_ids(self, date) -> SqlResult:
        return self.run("SELECT district_id FROM fact_daily WHERE date = ? AND hotspot", [date])

    def district_profile(self, district_id: int) -> SqlResult:
        sql = """
SELECT district, state, population,
       MAX(total_cases)                         AS total_cases,
       SUM(deaths)                              AS deaths,
       MAX(incidence_7d)                        AS peak_incidence,
       ARG_MAX(date, incidence_7d)              AS peak_date,
       SUM(CAST(hotspot AS INTEGER))            AS days_as_hotspot,
       MIN(CASE WHEN hotspot THEN date END)     AS first_hotspot,
       ARG_MAX(incidence_7d, date)              AS incidence_last_day,
       AVG(mobility_value)                      AS avg_mobility
FROM v_daily WHERE district_id = ?
GROUP BY district, state, population"""
        return self.run(sql, [district_id])

    def state_profile(self, state: str) -> SqlResult:
        sql = """
WITH d AS (
    SELECT date, SUM(total_cases) AS total_cases, SUM(deaths) AS deaths,
           SUM(cases_7d) * 100000.0 / SUM(population) AS incidence,
           SUM(CAST(hotspot AS INTEGER)) AS hotspots, SUM(population) AS population
    FROM v_daily WHERE state = ? GROUP BY date)
SELECT ? AS state, MAX(population) AS population, MAX(total_cases) AS total_cases,
       SUM(deaths) AS deaths, MAX(incidence) AS peak_incidence,
       ARG_MAX(date, incidence) AS peak_date, MAX(hotspots) AS max_hotspot_districts,
       ARG_MAX(incidence, date) AS incidence_last_day,
       (SELECT COUNT(*) FROM dim_district JOIN dim_state USING (state_id) WHERE state = ?) AS n_districts
FROM d"""
        return self.run(sql, [state, state, state])

    def districts(self) -> pd.DataFrame:
        return self.run("""
SELECT d.district_id, d.district AS name, s.state, d.population, d.lat, d.lon
FROM dim_district d JOIN dim_state s USING (state_id)""").df

    # ---- used only by the benchmark --------------------------------------
    def k_hop_recursive(self, district_id: int, k: int) -> SqlResult:
        sql = """
WITH RECURSIVE walk(id, hops) AS (
    SELECT ?, 0
    UNION ALL
    SELECT b.b, w.hops + 1
    FROM walk w JOIN district_border b ON b.a = w.id
    WHERE w.hops < ?
)
SELECT w.id AS district_id, d.district AS name, MIN(w.hops) AS hops
FROM walk w JOIN dim_district d ON d.district_id = w.id
WHERE w.id <> ?
GROUP BY w.id, d.district ORDER BY hops, name"""
        return self.run(sql, [district_id, k, district_id])


_sql: SqlEngine | None = None


def get_sql() -> SqlEngine:
    global _sql
    if _sql is None:
        _sql = SqlEngine()
    return _sql
