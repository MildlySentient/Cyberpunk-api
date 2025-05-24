import os
import logging
import logging.config
import math
import yaml
from typing import List, Dict, Any, Optional
import duckdb
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseSettings

# ----- CONFIGURATION -----
class Settings(BaseSettings):
    cors_origins: List[str] = ["*"]
    data_dir: str = os.path.dirname(__file__)
    logging_config: str = os.path.join(os.path.dirname(__file__), 'logging.yaml')
    class Config:
        env_file = ".env"
settings = Settings()

def load_yaml_config(filepath: str) -> Optional[Dict[str, Any]]:
    try:
        with open(filepath, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Failed to load YAML config '{filepath}': {e}")
        return None

config = load_yaml_config(settings.logging_config)
if config:
    logging.config.dictConfig(config)
else:
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cyberpunk_api")

# ----- DATA LOADING -----
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

duckdb_conn = duckdb.connect(database=":memory:")

def load_all_files():
    for fname in REQUIRED_FILES:
        fpath = os.path.join(settings.data_dir, fname)
        if not os.path.isfile(fpath):
            logger.warning(f"{fname} not found, skipping.")
            continue
        logger.info(f"Loading {fname} into DuckDB as '{fname.replace('.tsv','')}'")
        df = pd.read_csv(fpath, sep='\t', dtype=str).fillna("")
        duckdb_conn.register(fname.replace('.tsv',''), df)
    logger.info("DuckDB tables loaded.")

# ----- SANITIZATION -----
def sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize(v) for v in obj]
    elif isinstance(obj, float):
        return 0.0 if not math.isfinite(obj) else obj
    elif pd.isna(obj):
        return ""
    return obj

# ----- APP -----
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    load_all_files()
    logger.info("Loaded all core files.")

@app.get("/lookup")
def lookup(query: str = "", role: Optional[str] = None, gender: Optional[str] = None):
    debug = {"query": query, "role": role, "gender": gender}
    try:
        tablename = "prebuilt_characters"
        params = []
        where_clauses = []

        # SMART ROLE/GENDER DETECTION FROM QUERY
        if query and (not role or not gender):
            roles = duckdb_conn.execute("SELECT DISTINCT lower(Role) FROM prebuilt_characters").fetchall()
            roles = {r[0] for r in roles}
            genders = duckdb_conn.execute("SELECT DISTINCT lower(Gender) FROM prebuilt_characters").fetchall()
            genders = {g[0] for g in genders}
            tokens = [t.lower() for t in query.strip().replace(",", " ").split()]
            role_token = next((t for t in tokens if t in roles), None)
            gender_token = next((t for t in tokens if t in genders), None)
            if role_token:
                role = role_token
            if gender_token:
                gender = gender_token
            debug["autodetected_role"] = role
            debug["autodetected_gender"] = gender

        if role:
            where_clauses.append("lower(Role) = ?")
            params.append(role.lower())
        if gender:
            where_clauses.append("lower(Gender) = ?")
            params.append(gender.lower())

        if not where_clauses and query:
            where_clauses.append(
                "(lower(Name) LIKE ? OR lower(Role) LIKE ? OR lower(Gender) LIKE ?)"
            )
            q_like = f"%{query.lower()}%"
            params.extend([q_like, q_like, q_like])

        sql = f"SELECT * FROM {tablename}"
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)
        sql += " LIMIT 10"

        logger.info(f"RUNNING SQL: {sql} PARAMS: {params}")
        debug["sql"] = sql
        debug["params"] = params

        results = duckdb_conn.execute(sql, tuple(params)).fetchdf()
        debug["n_results"] = len(results)
        return {"debug": debug, "rows": sanitize(results.to_dict(orient="records"))}

    except Exception as e:
        logger.exception("Error during lookup")
        debug["error"] = str(e)
        return {"debug": debug, "error": str(e)}

@app.get("/sanity")
def sanity():
    try:
        df = duckdb_conn.execute("SELECT Name, Role, Gender FROM prebuilt_characters").fetchdf()
        return {"rows": sanitize(df.to_dict(orient="records"))}
    except Exception as e:
        return {"error": str(e)}