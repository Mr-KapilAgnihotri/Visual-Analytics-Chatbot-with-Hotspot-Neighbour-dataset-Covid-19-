"""
Step 2 of the ETL pipeline: reconstruct WHICH districts border each other.

The idea (a good viva talking point)
------------------------------------
The CSV never lists a district's neighbours, but for every district and every
day it gives Neighbor_Total_Cases = the SUM of Total_Cases of its bordering
districts. So for district i we look for a set S of nearby districts such that

        sum_{j in S} total_cases[j, t]  ==  nb_total_cases[i, t]   for EVERY day t

That is a 0/1 (binary) choice per candidate district, i.e. a small Mixed-Integer
Linear Programme (MILP). With 364 days of data the answer is essentially
unique, because a wrong set of districts would have to match 364 different
daily totals by pure coincidence.

We then check the result against the four other neighbour columns
(new cases, 7-day cases, deaths) and make the graph symmetric
(if A borders B, B borders A).

Run:  python -m etl.build_adjacency     (writes data/adjacency.json)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

ROOT = Path(__file__).resolve().parents[1]
CLEAN_PARQUET = ROOT / "data" / "clean.parquet"
ADJ_JSON = ROOT / "data" / "adjacency.json"
N_CANDIDATES = 30          # search among the 30 geographically nearest districts


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def solve_one(A: np.ndarray, b: np.ndarray):
    """Find binary x minimising |A x - b| (L1). Exact match => error 0."""
    n_days, n_cand = A.shape
    # variables: x (n_cand, binary) + slack e (n_days, >=0);  A x - b = e_plus - e_minus
    # we use |A x - b| <= e   ->   A x - e <= b   and   -A x - e <= -b
    c = np.concatenate([np.zeros(n_cand), np.ones(n_days)])
    I = np.eye(n_days)
    cons = LinearConstraint(
        np.vstack([np.hstack([A, -I]), np.hstack([-A, -I])]),
        -np.inf,
        np.concatenate([b, -b]),
    )
    integrality = np.concatenate([np.ones(n_cand), np.zeros(n_days)])
    bounds = Bounds(np.zeros(n_cand + n_days), np.concatenate([np.ones(n_cand), np.full(n_days, np.inf)]))
    res = milp(c, constraints=cons, integrality=integrality, bounds=bounds,
               options={"time_limit": 30})
    if res.x is None:
        return None, np.inf
    x = np.round(res.x[:n_cand]).astype(bool)
    return x, float(np.abs(A[:, x].sum(axis=1) - b).sum())


# ---------------------------------------------------------------------------
# Berlin special case.
# The source dataset treated Berlin as ONE city when it computed neighbour sums:
# every borough's Neighbor_Total_Cases equals the total of all 12 boroughs, and
# a Brandenburg district next to Berlin counts the whole city as one neighbour.
# So the solver sees Berlin as a single pseudo-district ("BERLIN"), and we then
# attach the 12 boroughs to each other and to Brandenburg using real geography.
# ---------------------------------------------------------------------------
BERLIN = -11000                      # pseudo-ID for "all of Berlin"
BERLIN_IDS = list(range(11001, 11013))
BERLIN_INTERNAL = {                  # borough -> neighbouring boroughs
    11001: [11002, 11003, 11004, 11007, 11012],
    11002: [11001, 11003, 11007, 11008, 11009, 11011],
    11003: [11001, 11002, 11011, 11012],
    11004: [11001, 11005, 11006, 11007, 11012],
    11005: [11004, 11006, 11012],
    11006: [11004, 11005, 11007],
    11007: [11001, 11002, 11004, 11006, 11008],
    11008: [11002, 11007, 11009],
    11009: [11002, 11008, 11010, 11011],
    11010: [11009, 11011],
    11011: [11002, 11003, 11009, 11010],
    11012: [11001, 11003, 11004, 11005],
}
BERLIN_OUTER = {                     # Brandenburg district -> boroughs it touches
    12054: [11005, 11006],           # Potsdam
    12060: [11003, 11010, 11011],    # Barnim
    12061: [11008, 11009],           # Dahme-Spreewald
    12063: [11005],                  # Havelland
    12064: [11009, 11010],           # Märkisch-Oderland
    12065: [11003, 11005, 11012],    # Oberhavel
    12067: [11009],                  # Oder-Spree
    12069: [11006],                  # Potsdam-Mittelmark
    12072: [11007],                  # Teltow-Fläming
}


def build(df: pd.DataFrame) -> dict:
    total = df.pivot(index="date", columns="district_id", values="total_cases").fillna(0)
    target = df.pivot(index="date", columns="district_id", values="nb_total_cases").fillna(0)
    info = df.drop_duplicates("district_id").set_index("district_id")
    ids = list(total.columns)
    solvable = [i for i in ids if i not in BERLIN_IDS]

    # candidate matrix: every non-Berlin district + one column for all of Berlin
    total_c = total[solvable].copy()
    total_c[BERLIN] = total[BERLIN_IDS].sum(axis=1)
    lat = info["lat"].to_dict(); lon = info["lon"].to_dict()
    lat[BERLIN] = info.loc[BERLIN_IDS, "lat"].mean(); lon[BERLIN] = info.loc[BERLIN_IDS, "lon"].mean()
    pool = solvable + [BERLIN]

    adj, errors = {}, {}
    for i in solvable:
        d = haversine_km(lat[i], lon[i], np.array([lat[p] for p in pool]), np.array([lon[p] for p in pool]))
        cand = [pool[k] for k in np.argsort(d) if pool[k] != i][:N_CANDIDATES]
        b = target[i].values.astype(float)
        if b.sum() == 0:                       # an island district with no neighbours
            adj[i], errors[i] = [], 0.0
            continue
        x, err = solve_one(total_c[cand].values.astype(float), b)
        adj[i] = [] if x is None else [c for c, keep in zip(cand, x) if keep]
        errors[i] = err

    # Berlin as a whole: check that the data agrees with "neighbours = whole city"
    berlin_err = float(np.abs(target[11001].values - total_c[BERLIN].values).sum())
    print(f"Berlin boroughs: neighbour sum equals whole-city total (error {berlin_err:.0f})")

    # symmetrise non-Berlin edges: keep an edge if either side found it
    edges = {tuple(sorted((a, b))) for a, ns in adj.items() for b in ns if b != BERLIN}
    asym = [(a, b) for a, b in edges if not (b in adj.get(a, []) and a in adj.get(b, []))]

    # attach Berlin boroughs from geography, only for districts the DATA says border Berlin
    data_says_berlin = {i for i, ns in adj.items() if BERLIN in ns}
    missing = data_says_berlin - set(BERLIN_OUTER)
    if missing:
        print("WARNING: data says these border Berlin but no geography entry:", missing)
    # Edges carry their provenance: "data" (recovered by the solver) or
    # "geography" (Berlin's real borders, which the source data never counted).
    source = {e: "data" for e in edges}
    for bid, ns in BERLIN_INTERNAL.items():
        for n in ns:
            source.setdefault(tuple(sorted((bid, n))), "geography")
    for did, boroughs in BERLIN_OUTER.items():
        for bz in boroughs:
            source.setdefault(tuple(sorted((did, bz))), "geography")
    edges = set(source)

    sym = {i: sorted({b for a, b in edges if a == i} | {a for a, b in edges if b == i}) for i in ids}
    exact = sum(1 for e in errors.values() if e == 0)
    print(f"Non-Berlin districts solved exactly: {exact}/{len(solvable)}")
    print(f"Districts the data says border Berlin: {sorted(info.loc[list(data_says_berlin), 'district'])}")
    print(f"Border edges: {len(edges)}   asymmetric (one-sided) edges: {len(asym)}")
    if asym:
        print("  one-sided:", [(info.at[a, 'district'], info.at[b, 'district']) for a, b in asym])
    bad = {k: v for k, v in errors.items() if v > 0}
    if bad:
        print("Districts with residual error:", {info.at[k, 'district']: round(v) for k, v in bad.items()})
    # Fold the temporary Eisenach node into Wartburgkreis (they are one unit)
    from etl.clean import EISENACH_TEMP_ID
    def fold(x):
        return 16063 if x == EISENACH_TEMP_ID else x
    folded: dict = {}
    for (a, b), src in source.items():
        a, b = fold(a), fold(b)
        if a != b:
            folded.setdefault(tuple(sorted((a, b))), src)
    source = folded
    ids = [i for i in ids if i != EISENACH_TEMP_ID]
    sym = {i: sorted({b for a, b in source if a == i} | {a for a, b in source if b == i}) for i in ids}
    print(f"After folding Eisenach into Wartburgkreis: {len(source)} border edges")

    return {"edges": [[int(a), int(b), source[(a, b)]] for a, b in sorted(source)],
            "adjacency": {str(k): [int(x) for x in v] for k, v in sym.items()},
            "data_neighbours": {str(k): [int(x) for x in v] for k, v in adj.items()},
            "residual_error": {str(k): v for k, v in errors.items()}}


def verify(df: pd.DataFrame, data_neighbours: dict) -> None:
    """Check the recovered neighbour sets against the other Neighbor_* columns
    (the solver only ever looked at nb_total_cases, so these are independent checks)."""
    cols = [("total_cases", "nb_total_cases"), ("new_cases", "nb_new_cases"),
            ("cases_7d", "nb_cases_7d"), ("deaths", "nb_deaths")]
    for src, nb in cols:
        piv = df.pivot(index="date", columns="district_id", values=src).fillna(0)
        piv[BERLIN] = piv[BERLIN_IDS].sum(axis=1)
        tgt = df.pivot(index="date", columns="district_id", values=nb).fillna(0)
        sets = {int(i): [int(j) for j in ns] for i, ns in data_neighbours.items()}
        sets.update({b: [BERLIN] for b in BERLIN_IDS})
        ok = sum(np.allclose(piv[ns].sum(axis=1).values, tgt[i].values) for i, ns in sets.items())
        print(f"  {nb:<15} matches for {ok}/{len(sets)} districts")


if __name__ == "__main__":
    from etl.clean import clean
    data = clean(merge_eisenach=False)
    result = build(data)
    print("Verification against every neighbour column:")
    verify(data, result["data_neighbours"])
    ADJ_JSON.write_text(json.dumps(result, indent=1))
    print(f"-> {ADJ_JSON}")
