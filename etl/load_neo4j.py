"""
Step 4 of the ETL pipeline: load the knowledge graph into Neo4j.

Uses batched UNWIND statements (one round-trip per node/relationship type
instead of one per row) and MERGE (create-if-missing), and wipes the previous
graph first, so the load is IDEMPOTENT.

Run:  python -m etl.load_neo4j     (Neo4j must be running; credentials in .env)
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.config import NEO4J_DATABASE, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER  # noqa: E402
from etl.graph_payload import build_payload  # noqa: E402

CONSTRAINTS = [
    "CREATE CONSTRAINT district_id IF NOT EXISTS FOR (d:District) REQUIRE d.id IS UNIQUE",
    "CREATE CONSTRAINT state_name IF NOT EXISTS FOR (s:State) REQUIRE s.name IS UNIQUE",
    "CREATE CONSTRAINT month_ym IF NOT EXISTS FOR (m:Month) REQUIRE m.ym IS UNIQUE",
    "CREATE INDEX district_name IF NOT EXISTS FOR (d:District) ON (d.name)",
]

LOAD_STEPS = [
    ("states", """
        UNWIND $rows AS r
        MERGE (s:State {name: r.name})
        SET s.population = r.population, s.n_districts = r.n_districts"""),
    ("months", """
        UNWIND $rows AS ym
        MERGE (:Month {ym: ym})"""),
    ("districts", """
        UNWIND $rows AS r
        MERGE (d:District {id: r.id})
        SET d.name = r.name, d.state = r.state, d.population = r.population,
            d.lat = r.lat, d.lon = r.lon,
            d.peak_incidence = r.peak_incidence, d.peak_date = date(r.peak_date),
            d.days_as_hotspot = r.days_as_hotspot,
            d.first_hotspot_date = CASE WHEN r.first_hotspot_date IS NULL THEN null
                                        ELSE date(r.first_hotspot_date) END,
            d.total_cases = r.total_cases, d.total_deaths = r.total_deaths
        WITH d, r
        MATCH (s:State {name: r.state})
        MERGE (d)-[:IN_STATE]->(s)"""),
    ("borders", """
        UNWIND $rows AS r
        MATCH (a:District {id: r.a}), (b:District {id: r.b})
        MERGE (a)-[x:BORDERS]-(b)
        SET x.source = r.source"""),
    ("state_borders", """
        UNWIND $rows AS r
        MATCH (a:State {name: r.a}), (b:State {name: r.b})
        MERGE (a)-[x:BORDERS]-(b)
        SET x.shared_borders = r.shared_borders"""),
    ("hotspot_months", """
        UNWIND $rows AS r
        MATCH (d:District {id: r.id}), (m:Month {ym: r.ym})
        MERGE (d)-[x:HOTSPOT_IN]->(m)
        SET x.days = r.days, x.peak = r.peak"""),
]


def load() -> None:
    from neo4j import GraphDatabase

    t0 = time.perf_counter()
    payload = build_payload()
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    with driver.session(database=NEO4J_DATABASE) as s:
        s.run("MATCH (n) WHERE n:District OR n:State OR n:Month DETACH DELETE n").consume()
        for c in CONSTRAINTS:
            s.run(c).consume()
        for key, cypher in LOAD_STEPS:
            s.run(cypher, rows=payload[key]).consume()
            print(f"  loaded {key:<15} {len(payload[key]):>5}")
        counts = s.run("""
            MATCH (n) WITH count(n) AS nodes
            MATCH ()-[r]->() RETURN nodes, count(r) AS rels""").single()
    driver.close()
    print(f"Neo4j loaded in {time.perf_counter() - t0:.1f}s: "
          f"{counts['nodes']} nodes, {counts['rels']} relationships")


if __name__ == "__main__":
    load()
