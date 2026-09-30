"""
PARITY TEST: proves the Cypher queries in Neo4j return exactly the same answers
as the reference NetworkX implementation.

Run after loading Neo4j:   python -m tests.check_neo4j_parity
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.disable(logging.WARNING)

from backend.graph_engine import Neo4jGraph, NetworkXGraph  # noqa: E402
from backend.sql_engine import get_sql  # noqa: E402


def norm(rows, keys):
    return sorted(tuple(r[k] for k in keys) for r in rows)


def main() -> int:
    neo, ref = Neo4jGraph(), NetworkXGraph()
    hot = get_sql().hotspot_ids("2020-12-01").df["district_id"].tolist()
    checks = [
        ("neighbours Köln", lambda g: norm(g.neighbours(5315).rows, ["id", "source"])),
        ("neighbours Berlin Mitte", lambda g: norm(g.neighbours(11001).rows, ["id", "source"])),
        ("k_hop München k=3", lambda g: norm(g.k_hop(9162, 3).rows, ["id", "hops"])),
        ("k_hop Jena k=5", lambda g: norm(g.k_hop(16053, 5).rows, ["id", "hops"])),
        ("shortest path length", lambda g: len(g.shortest_path(1001, 9262).rows)),
        ("most connected top 10", lambda g: [r["degree"] for r in g.most_connected(10).rows]),
        ("most connected in Sachsen", lambda g: norm(g.most_connected(5, "Sachsen").rows, ["id", "degree"])),
        ("state borders Hessen", lambda g: norm(g.state_borders("Hessen").rows, ["state", "shared_borders"])),
        ("edges among hotspots", lambda g: norm(g.edges_among(hot).rows, ["a", "b"])),
        ("surrounded hotspots", lambda g: norm(g.surrounded(hot).rows, ["id", "n_neighbours"])),
        ("isolated hotspots", lambda g: norm(g.isolated(hot).rows, ["id"])),
        ("cluster sizes", lambda g: [c["size"] for c in g.clusters(hot).rows]),
        ("all edges", lambda g: len(g.all_edges().rows)),
    ]
    failures = 0
    for name, fn in checks:
        a, b = fn(neo), fn(ref)
        ok = a == b
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            print(f"      neo4j   : {str(a)[:200]}\n      networkx: {str(b)[:200]}")
    print(f"\n{len(checks) - failures}/{len(checks)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
