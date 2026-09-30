# Visual Analytics Chatbot: COVID-19 in Germany, 2020

Ask questions in plain English and get **a chart plus a written explanation**.
Behind the scenes, each question is answered by the engine that suits it best:

| Question shape | Engine | Example |
|---|---|---|
| Aggregations over the daily time series | **DuckDB** (relational, SQL) | "Top 10 districts by incidence on 15 November" |
| Relationships between districts (borders) | **Neo4j** (knowledge graph, Cypher) | "Shortest path from Flensburg to Passau" |
| Both: graph finds *who*, SQL measures *how much* | **Hybrid** | "Districts within 2 borders of Berlin Mitte with incidence above 100" |

The language model (**Groq**) only reads the question into a small, validated form
and puts computed facts into prose; it never writes queries and never invents numbers.

---

## 1. One-time setup (Windows, PowerShell)

```powershell
cd "D:\VA chatbot"
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
copy .env.example .env                # then open .env in an editor
```

### Neo4j (local, free)
1. Download **Neo4j Desktop** from https://neo4j.com/download/ and install it.
2. Create a local instance/database (e.g. name `va-chatbot`), set a password, and **start** it.
3. Put that password into `.env` as `NEO4J_PASSWORD=...`. The default URI `bolt://localhost:7687` is correct.

### Groq (free LLM)
1. Sign up at https://console.groq.com, go to **API Keys**, and create a key.
2. Put it into `.env` as `GROQ_API_KEY=...`.

Both are optional: without Neo4j the app uses an identical in-memory graph (NetworkX),
and without a Groq key it uses its rule-based parser and template text.

## 2. Build the databases (once, about 1 minute)

```powershell
python -m etl.run_all                  # clean -> borders -> DuckDB -> Neo4j
python -m tests.check_neo4j_parity     # proves Neo4j answers == reference answers
python -m pytest -q                    # 47 automated tests
```
(`python -m etl.run_all --no-neo4j` if Neo4j is not running yet.
Stop the API and the chat UI before re-running the ETL, because DuckDB allows only one writer.)

## 3. Run the chatbot (two terminals, venv activated in both)

```powershell
uvicorn backend.main:app --reload      # terminal 1: API  -> http://127.0.0.1:8000/docs
streamlit run frontend/app.py          # terminal 2: chat -> http://localhost:8501
```
The chat UI also works without terminal 1: it then answers in-process.

## 4. Benchmark (which engine is faster for what)

```powershell
python -m benchmarks.run_benchmark --with-observations
```
Writes `benchmarks/results.csv` and an interactive `benchmarks/results.html`.

---

## Architecture

```
 question ─► [1] NLU ──────────────► QueryPlan {intent, metric, places, dates, n, hops…}
             entities.py (aliases, fuzzy match, dates)  +  nlu.py (rules, then Groq JSON, validated)
          ─► [2] Router (catalog.INTENTS) ─► SQL | GRAPH | HYBRID
          ─► [3] Execute
                 sql_engine.py   DuckDB, parameterised SQL templates
                 graph_engine.py Neo4j Cypher  (NetworkX fallback, same answers)
          ─► [4] Fuse           pandas merge on district_id
          ─► [5] Visualise      charts.py → Plotly (bar, donut, line, scatter, map, network-on-map)
          ─► [6] Narrate        narrator.py: facts computed in Python → Groq writes prose
                                → number-grounding check → fallback template if it fails
          ─► Answer {text, chart, table, KPIs, exact queries, timings}
```

### Data model
**DuckDB (star schema):** `dim_state`, `dim_district`, `fact_daily` (149,592 rows), `district_border`, view `v_daily`.

**Neo4j (knowledge graph):**
```
(:District {id, name, state, population, lat, lon, peak_incidence, days_as_hotspot, …})
(:State {name, population}) (:Month {ym})
(:District)-[:IN_STATE]->(:State)
(:District)-[:BORDERS {source}]-(:District)      1,080 borders
(:State)-[:BORDERS {shared_borders}]-(:State)
(:District)-[:HOTSPOT_IN {days, peak}]->(:Month)
```

### Data-quality work done in the ETL (see `etl/clean.py`, `etl/build_adjacency.py`)
* **Borders were not in the CSV.** They were reconstructed exactly from the `Neighbor_*`
  sums by solving a small integer programme per district. They are verified against all four
  neighbour columns on all 364 days (412/412 exact) and are fully symmetric.
* **21 city/rural-district name swaps corrected** (e.g. key 09162 is Munich *city*, 1.49 M people).
* **Eisenach + Wartburgkreis** carried identical case data and were merged into one unit.
* **Berlin's 12 boroughs** are named individually; the source summed Berlin as one city,
  so the borough borders come from real geography (marked `source: "geography"`).
* **Mobility is recorded per state**, so mobility correlations are computed across the 16 states.

## Project layout
```
data/        raw CSV, derived files (clean.parquet, adjacency.json, covid.duckdb)
etl/         clean.py, build_adjacency.py, load_duckdb.py, graph_payload.py, load_neo4j.py, run_all.py
backend/     config, catalog, entities, nlu, llm, sql_engine, graph_engine, charts, narrator, orchestrator, main (API)
frontend/    app.py (Streamlit) + static/topojson (map outlines, served locally)
benchmarks/  run_benchmark.py
tests/       test_pipeline.py, check_neo4j_parity.py
```
