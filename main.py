import os
import logging
import duckdb
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseSettings
from typing import List, Optional

# ----------- CONFIGURATION -----------
class Settings(BaseSettings):
    cors_origins: List[str] = ["*"]
    data_dir: str = os.path.dirname(__file__)
    class Config:
        env_file = ".env"

settings = Settings()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cyberpunk_api")

# ----------- DUCKDB SETUP -----------
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

duckdb_conn = duckdb.connect(database=':memory:', read_only=False)

def load_tsv_to_duckdb():
    for fname in REQUIRED_FILES:
        fpath = os.path.join(settings.data_dir, fname)
        if not os.path.isfile(fpath):
            logger.warning(f"{fname} not found, skipping.")
            continue
        table = os.path.splitext(fname)[0]
        # DuckDB can read TSV directly
        logger.info(f"Loading {fname} into DuckDB as '{table}'")
        duckdb_conn.execute(f"""
            CREATE OR REPLACE TABLE {table} AS 
            SELECT * FROM read_csv_auto('{fpath}', sep='\t', header=True, NULLSTR=''); 
        """)

load_tsv_to_duckdb()

# ----------- FASTAPI SETUP -----------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    # reload on startup for hot reloads/local dev
    load_tsv_to_duckdb()
    logger.info("DuckDB tables loaded.")

@app.get("/lookup")
def lookup(query: str, role: Optional[str] = None, gender: Optional[str] = None):
    # For this example: search prebuilt_characters for a matching role/gender/name
    where_clauses = []
    params = {}
    if role:
        where_clauses.append("lower(Role) = lower(:role)")
        params['role'] = role
    if gender:
        where_clauses.append("lower(Gender) = lower(:gender)")
        params['gender'] = gender
    if query:
        # partial name or text search
        where_clauses.append("(lower(Name) LIKE lower(:q) OR lower(Role) LIKE lower(:q) OR lower(Gender) LIKE lower(:q))")
        params['q'] = f"%{query}%"

    sql = "SELECT * FROM prebuilt_characters"
    if where_clauses:
        sql += " WHERE " + " AND ".join(where_clauses)
    sql += " LIMIT 10"

    logger.info(f"RUNNING SQL: {sql} PARAMS: {params}")
    results = duckdb_conn.execute(sql, params).fetchdf()
    # Convert to dict for FastAPI
    return {"rows": results.to_dict(orient="records")}

@app.get("/sql")
def sql_endpoint(sql: str):
    try:
        df = duckdb_conn.execute(sql).fetchdf()
        return {"rows": df.to_dict(orient="records")}
    except Exception as e:
        return {"error": str(e)}

@app.get("/sanity")
def sanity():
    # Print all roles/genders combos for prebuilt_characters
    try:
        df = duckdb_conn.execute("SELECT Name, Role, Gender FROM prebuilt_characters").fetchdf()
        return {"rows": df.to_dict(orient="records")}
    except Exception as e:
        return {"error": str(e)}