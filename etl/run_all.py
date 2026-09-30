"""
Runs the whole ETL pipeline in order:

    1. clean            raw CSV            -> data/clean.parquet
    2. build_adjacency  neighbour sums     -> data/adjacency.json   (skipped if it exists; ~1 min)
    3. load_duckdb      clean + borders    -> data/covid.duckdb
    4. load_neo4j       DuckDB summaries   -> Neo4j                 (skip with --no-neo4j)

Usage:
    python -m etl.run_all                  # everything
    python -m etl.run_all --no-neo4j       # when Neo4j is not running yet
    python -m etl.run_all --rebuild-borders
"""
import argparse
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.config import ADJ_JSON, CLEAN_PARQUET  # noqa: E402
from etl import build_adjacency, clean, load_duckdb  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-neo4j", action="store_true")
    ap.add_argument("--rebuild-borders", action="store_true")
    args = ap.parse_args()

    print("[1/4] Cleaning raw CSV")
    df = clean.clean()
    df.to_parquet(CLEAN_PARQUET, index=False)
    print(f"      {len(df):,} rows, {df['district_id'].nunique()} districts")

    if args.rebuild_borders or not Path(ADJ_JSON).exists():
        print("[2/4] Reconstructing district borders (about a minute)")
        unmerged = clean.clean(merge_eisenach=False)
        result = build_adjacency.build(unmerged)
        build_adjacency.verify(unmerged, result["data_neighbours"])
        Path(ADJ_JSON).write_text(json.dumps(result, indent=1))
    else:
        print("[2/4] Borders already built (use --rebuild-borders to redo)")

    print("[3/4] Loading DuckDB")
    load_duckdb.load()

    if args.no_neo4j:
        print("[4/4] Skipped Neo4j (the app will use the NetworkX graph instead)")
    else:
        print("[4/4] Loading Neo4j")
        from etl import load_neo4j
        load_neo4j.load()
    print("ETL complete.")


if __name__ == "__main__":
    main()
