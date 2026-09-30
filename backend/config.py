"""
Central configuration. Every setting can be overridden in the `.env` file
(copy `.env.example` to `.env` and fill in your values).
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "hotspot_neighbors.csv"
CLEAN_PARQUET = DATA_DIR / "clean.parquet"
ADJ_JSON = DATA_DIR / "adjacency.json"
DUCKDB_PATH = Path(os.getenv("DUCKDB_PATH", DATA_DIR / "covid.duckdb"))

# --- Knowledge graph -------------------------------------------------------
# GRAPH_BACKEND = "neo4j"    -> use the Neo4j server (falls back to NetworkX if unreachable)
# GRAPH_BACKEND = "networkx" -> in-process graph, no server needed
GRAPH_BACKEND = os.getenv("GRAPH_BACKEND", "neo4j").lower()
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# --- LLM (Groq, OpenAI-compatible API) --------------------------------------
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_API_KEY = os.getenv("GROQ_API_KEY", os.getenv("LLM_API_KEY", ""))
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "20"))
# set USE_LLM=false to run fully offline with the rule-based parser
USE_LLM = os.getenv("USE_LLM", "true").lower() == "true" and bool(LLM_API_KEY)

# --- API --------------------------------------------------------------------
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# --- Dataset facts ------------------------------------------------------------
HOTSPOT_THRESHOLD = 75          # 7-day incidence per 100k at/above which Hotspot = 1
FIRST_DATE = "2020-01-03"
LAST_DATE = "2020-12-31"
