"""
The KNOWLEDGE-GRAPH engine.

Two interchangeable backends expose the SAME methods and return the SAME answers:

  * Neo4jGraph    - the real knowledge graph, queried in Cypher (primary)
  * NetworkXGraph - an in-process copy of the same graph (fallback when the Neo4j
                    server is not running, and a reference for the parity test)

Every method returns a GraphResult holding the rows, the exact query that was
run (shown in the UI so the examiner can see the Cypher) and its runtime.

Why these questions belong to the graph: each one follows BORDERS relationships.
In a graph database a node stores direct pointers to its relationships
("index-free adjacency"), so hopping to a neighbour costs the same whether the
database holds 400 districts or 4 million - the work grows with the part of the
graph you touch, not with the table size.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import networkx as nx

from backend import config

log = logging.getLogger(__name__)
MAX_HOPS = 10          # the NLU limits user questions to 6; the benchmark goes further


@dataclass
class GraphResult:
    rows: list
    query: str
    ms: float
    language: str = "Cypher"
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Cypher queries (one per question type)
# ---------------------------------------------------------------------------
CYPHER = {
    "neighbours": """
MATCH (d:District {id: $id})-[b:BORDERS]-(n:District)
RETURN n.id AS id, n.name AS name, n.state AS state, b.source AS source
ORDER BY name""",
    # the upper bound of a variable-length pattern cannot be a parameter, so the
    # validated integer k is formatted in (see _k_hop_cypher)
    "k_hop": """
MATCH (d:District {id: $id}), (n:District)
WHERE n <> d
MATCH p = shortestPath((d)-[:BORDERS*..%d]-(n))
RETURN n.id AS id, n.name AS name, n.state AS state, length(p) AS hops
ORDER BY hops, name""",
    "shortest_path": """
MATCH (a:District {id: $a}), (b:District {id: $b})
MATCH p = shortestPath((a)-[:BORDERS*..40]-(b))
RETURN [n IN nodes(p) | {id: n.id, name: n.name, state: n.state}] AS path""",
    "most_connected": """
MATCH (d:District)
WHERE $state IS NULL OR d.state = $state
RETURN d.id AS id, d.name AS name, d.state AS state,
       COUNT { (d)-[:BORDERS]-(:District) } AS degree
ORDER BY degree DESC, name
LIMIT $n""",
    "state_borders": """
MATCH (s:State {name: $state})-[b:BORDERS]-(o:State)
RETURN o.name AS state, b.shared_borders AS shared_borders
ORDER BY shared_borders DESC, state""",
    "edges_among": """
MATCH (a:District)-[:BORDERS]-(b:District)
WHERE a.id IN $ids AND b.id IN $ids AND a.id < b.id
RETURN a.id AS a, b.id AS b""",
    "surrounded": """
MATCH (d:District) WHERE d.id IN $ids
WITH d, [(d)-[:BORDERS]-(n:District) | n.id] AS nbrs
WHERE size(nbrs) > 0 AND all(x IN nbrs WHERE x IN $ids)
RETURN d.id AS id, d.name AS name, d.state AS state, size(nbrs) AS n_neighbours
ORDER BY n_neighbours DESC, name""",
    "isolated": """
MATCH (d:District) WHERE d.id IN $ids
WITH d, [(d)-[:BORDERS]-(n:District) | n.id] AS nbrs
WHERE none(x IN nbrs WHERE x IN $ids)
RETURN d.id AS id, d.name AS name, d.state AS state, size(nbrs) AS n_neighbours
ORDER BY name""",
    "all_edges": """
MATCH (a:District)-[b:BORDERS]-(c:District) WHERE a.id < c.id
RETURN a.id AS a, c.id AS b, b.source AS source""",
}


def _components(ids: list[int], edges: list[tuple[int, int]]) -> list[list[int]]:
    """Connected components of the sub-graph induced by `ids` (union-find)."""
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    groups: dict = {}
    for i in ids:
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=len, reverse=True)


class BaseGraph:
    """Shared logic. Sub-classes implement `_run(name, **params)`."""
    backend = "base"
    language = "Cypher"

    def _timed(self, name: str, query_text: str, **params) -> GraphResult:
        t0 = time.perf_counter()
        rows = self._run(name, **params)
        return GraphResult(rows, query_text, (time.perf_counter() - t0) * 1000, self.language)

    # ---- public API: identical for both backends -------------------------
    def neighbours(self, district_id: int) -> GraphResult:
        return self._timed("neighbours", self._text("neighbours"), id=district_id)

    def k_hop(self, district_id: int, k: int) -> GraphResult:
        k = max(1, min(int(k), MAX_HOPS))
        return self._timed("k_hop", self._text("k_hop", k), id=district_id, k=k)

    def shortest_path(self, a: int, b: int) -> GraphResult:
        return self._timed("shortest_path", self._text("shortest_path"), a=a, b=b)

    def most_connected(self, n: int = 10, state: str | None = None) -> GraphResult:
        return self._timed("most_connected", self._text("most_connected"), n=n, state=state)

    def state_borders(self, state: str) -> GraphResult:
        return self._timed("state_borders", self._text("state_borders"), state=state)

    def edges_among(self, ids: list[int]) -> GraphResult:
        return self._timed("edges_among", self._text("edges_among"), ids=list(map(int, ids)))

    def surrounded(self, ids: list[int]) -> GraphResult:
        return self._timed("surrounded", self._text("surrounded"), ids=list(map(int, ids)))

    def isolated(self, ids: list[int]) -> GraphResult:
        return self._timed("isolated", self._text("isolated"), ids=list(map(int, ids)))

    def all_edges(self) -> GraphResult:
        return self._timed("all_edges", self._text("all_edges"))

    def clusters(self, ids: list[int]) -> GraphResult:
        """Groups of mutually bordering districts among `ids`.
        The graph returns the borders inside the set; grouping them into
        connected components is a union-find in Python (Neo4j's Graph Data
        Science library offers the same as gds.wcc if installed)."""
        er = self.edges_among(ids)
        comps = _components(list(map(int, ids)), [(r["a"], r["b"]) for r in er.rows])
        return GraphResult([{"cluster": i + 1, "ids": c, "size": len(c)} for i, c in enumerate(comps)],
                           er.query + "\n// + connected components (union-find) in Python",
                           er.ms, self.language, {"edges": er.rows})

    def _text(self, name: str, k: int | None = None) -> str:
        q = CYPHER[name].strip()
        return q % k if k is not None else q

    def status(self) -> dict:
        return {"backend": self.backend}


# ---------------------------------------------------------------------------
# Neo4j backend
# ---------------------------------------------------------------------------
class Neo4jGraph(BaseGraph):
    backend = "neo4j"
    language = "Cypher"

    def __init__(self):
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(config.NEO4J_URI,
                                           auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
        self.driver.verify_connectivity()
        n = self._query("MATCH (d:District) RETURN count(d) AS n")[0]["n"]
        if n == 0:
            raise RuntimeError("Neo4j is running but the graph is empty - run: python -m etl.load_neo4j")
        self.n_districts = n

    def _query(self, cypher: str, **params) -> list[dict]:
        records, _, _ = self.driver.execute_query(cypher, params, database_=config.NEO4J_DATABASE)
        return [r.data() for r in records]

    def _run(self, name: str, **p) -> list[dict]:
        if name == "k_hop":
            return self._query(CYPHER["k_hop"] % p["k"], id=p["id"])
        if name == "shortest_path":
            if p["a"] == p["b"]:
                return self._query("MATCH (a:District {id:$a}) RETURN a.id AS id, a.name AS name, a.state AS state",
                                   a=p["a"])
            rows = self._query(CYPHER["shortest_path"], a=p["a"], b=p["b"])
            return rows[0]["path"] if rows else []
        return self._query(CYPHER[name], **p)

    def status(self) -> dict:
        return {"backend": "neo4j", "uri": config.NEO4J_URI, "districts": self.n_districts}


# ---------------------------------------------------------------------------
# NetworkX backend (same knowledge, held in memory)
# ---------------------------------------------------------------------------
class NetworkXGraph(BaseGraph):
    backend = "networkx"
    language = "NetworkX (Python)"

    PY = {
        "neighbours": "G.neighbors(id)",
        "k_hop": "nx.single_source_shortest_path_length(G, id, cutoff=%d)",
        "shortest_path": "nx.shortest_path(G, a, b)",
        "most_connected": "sorted(G.degree, key=degree, reverse=True)[:n]",
        "state_borders": "S.neighbors(state)   # S = state-level border graph",
        "edges_among": "G.subgraph(ids).edges()",
        "surrounded": "[d for d in ids if all(n in ids for n in G[d])]",
        "isolated": "[d for d in ids if not any(n in ids for n in G[d])]",
        "all_edges": "G.edges(data='source')",
    }

    def __init__(self, payload: dict | None = None):
        if payload is None:
            from etl.graph_payload import build_payload
            payload = build_payload()
        self.G = nx.Graph()
        for d in payload["districts"]:
            self.G.add_node(d["id"], **d)
        for e in payload["borders"]:
            self.G.add_edge(e["a"], e["b"], source=e["source"])
        self.S = nx.Graph()
        for s in payload["state_borders"]:
            self.S.add_edge(s["a"], s["b"], shared_borders=s["shared_borders"])
        self.n_districts = self.G.number_of_nodes()

    def _text(self, name: str, k: int | None = None) -> str:
        q = self.PY[name]
        return q % k if k is not None else q

    def _node(self, i: int) -> dict:
        n = self.G.nodes[i]
        return {"id": i, "name": n["name"], "state": n["state"]}

    def _run(self, name: str, **p) -> list[dict]:
        G = self.G
        if name == "neighbours":
            return sorted([{**self._node(n), "source": G.edges[p["id"], n]["source"]} for n in G[p["id"]]],
                          key=lambda r: r["name"])
        if name == "k_hop":
            dist = nx.single_source_shortest_path_length(G, p["id"], cutoff=p["k"])
            return sorted([{**self._node(n), "hops": h} for n, h in dist.items() if n != p["id"]],
                          key=lambda r: (r["hops"], r["name"]))
        if name == "shortest_path":
            try:
                return [self._node(n) for n in nx.shortest_path(G, p["a"], p["b"])]
            except nx.NetworkXNoPath:
                return []
        if name == "most_connected":
            rows = [{**self._node(n), "degree": deg} for n, deg in G.degree
                    if p.get("state") is None or G.nodes[n]["state"] == p["state"]]
            return sorted(rows, key=lambda r: (-r["degree"], r["name"]))[: p["n"]]
        if name == "state_borders":
            if p["state"] not in self.S:
                return []
            rows = [{"state": o, "shared_borders": self.S.edges[p["state"], o]["shared_borders"]}
                    for o in self.S[p["state"]]]
            return sorted(rows, key=lambda r: (-r["shared_borders"], r["state"]))
        ids = set(p.get("ids", []))
        if name == "edges_among":
            return [{"a": min(a, b), "b": max(a, b)} for a, b in G.subgraph(ids).edges()]
        if name == "surrounded":
            rows = [{**self._node(d), "n_neighbours": G.degree[d]} for d in ids
                    if d in G and G.degree[d] > 0 and all(n in ids for n in G[d])]
            return sorted(rows, key=lambda r: (-r["n_neighbours"], r["name"]))
        if name == "isolated":
            rows = [{**self._node(d), "n_neighbours": G.degree[d]} for d in ids
                    if d in G and not any(n in ids for n in G[d])]
            return sorted(rows, key=lambda r: r["name"])
        if name == "all_edges":
            return [{"a": min(a, b), "b": max(a, b), "source": s} for a, b, s in G.edges(data="source")]
        raise ValueError(name)

    def status(self) -> dict:
        return {"backend": "networkx", "districts": self.n_districts}


# ---------------------------------------------------------------------------
_graph: BaseGraph | None = None
_fallback_reason: str | None = None


def get_graph() -> BaseGraph:
    """Return the configured graph backend, falling back to NetworkX if Neo4j is unreachable."""
    global _graph, _fallback_reason
    if _graph is None:
        if config.GRAPH_BACKEND == "neo4j":
            try:
                _graph = Neo4jGraph()
                log.info("Knowledge graph: Neo4j at %s", config.NEO4J_URI)
            except Exception as exc:  # noqa: BLE001 - any connection problem -> fallback
                first_line = str(exc).splitlines()[0][:160] if str(exc) else ""
                _fallback_reason = (f"Neo4j not reachable at {config.NEO4J_URI} ({exc.__class__.__name__}: "
                                    f"{first_line}). Using the in-memory NetworkX copy of the same graph.")
                log.warning(_fallback_reason)
        if _graph is None:
            _graph = NetworkXGraph()
    return _graph


def graph_status() -> dict:
    g = get_graph()
    return {**g.status(), "fallback_reason": _fallback_reason}
