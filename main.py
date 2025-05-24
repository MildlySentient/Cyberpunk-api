import os
import logging
import duckdb
import pandas as pd
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# --- Logging setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cyberpunk_api")

# --- File configuration ---
REQUIRED_FILES = [
    "Index.tsv",
    "prebuilt_characters.tsv",
    "services_core.tsv",
    "plugins_core.tsv",
    "campaigns_core.tsv",
    "corporate_core.tsv",
    "encounters_core.tsv",
    "setting_core.tsv",
    "weapons_core.tsv",
    "assets_core.tsv",
    "system_core.tsv",
    "roles_core.tsv",
    "gear_core.tsv",
    "npc_core.tsv",
]
DATA_DIR = os.path.dirname(__file__)

# --- DuckDB setup ---
duckdb_conn = duckdb.connect(database=':memory:')
def load_tsvs_to_duckdb():
    for fname in REQUIRED_FILES:
        fpath = os.path.join(DATA_DIR, fname)
        tablename = os.path.splitext(fname)[0].replace('.', '_')
        if not os.path.isfile(fpath):
            logger.warning(f"{fname} not found, skipping.")
            continue
        logger.info(f"Loading {fname} into DuckDB as '{tablename}'")
        # If the table already exists, replace it
        try:
            duckdb_conn.execute(f"DROP TABLE IF EXISTS {tablename}")
            duckdb_conn.execute(
                f"CREATE TABLE {tablename} AS SELECT * FROM read_csv_auto('{fpath}', delim='\t', header=True, IGNORE_ERRORS=TRUE)"
            )
        except Exception as e:
            logger.error(f"Failed to load {fname}: {e}")
    logger.info("DuckDB tables loaded.")

# --- FastAPI setup ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    load_tsvs_to_duckdb()

@app.get("/lookup")
def lookup(
    query: str = "",
    role: Optional[str] = None,
    gender: Optional[str] = None
):
    debug = {"query": query, "role": role, "gender": gender}
    try:
        tablename = "prebuilt_characters"
        where_clauses = []
        params = []
        # Role filter
        if role:
            where_clauses.append("lower(Role) = lower(?)")
            params.append(role)
        # Gender filter
        if gender:
            where_clauses.append("lower(Gender) = lower(?)")
            params.append(gender)
        # Query (matches Name, Role, Gender)
        if query:
            where_clauses.append(
                "(lower(Name) LIKE lower(?) OR lower(Role) LIKE lower(?) OR lower(Gender) LIKE lower(?))"
            )
            q_like = f"%{query}%"
            params.extend([q_like, q_like, q_like])

        sql = f"SELECT * FROM {tablename}"
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)
        sql += " LIMIT 10"

        logger.info(f"RUNNING SQL: {sql} PARAMS: {params}")
        debug["sql"] = sql
        debug["params"] = params

        # DuckDB expects params as a tuple
        results = duckdb_conn.execute(sql, tuple(params)).fetchdf()
        debug["n_results"] = len(results)
        return {
            "debug": debug,
            "rows": results.to_dict(orient="records")
        }
    except Exception as e:
        logger.exception("Error during lookup")
        debug["error"] = str(e)
        return {"debug": debug, "error": str(e)}

@app.get("/sanity")
def sanity():
    try:
        sql = "SELECT Name, Role, Gender FROM prebuilt_characters"
        rows = duckdb_conn.execute(sql).fetchdf()
        return {"rows": rows.to_dict(orient="records")}
    except Exception as e:
        return {"error": str(e)}