import os
import logging
import logging.config
import math
import yaml
import re
from typing import List, Dict, Any, Optional, Set, Tuple
import pandas as pd
import duckdb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseSettings

# ----------- CONFIGURATION -----------
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

# ----------- DATA FILES TO LOAD -----------
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

def load_duckdb_tables():
    for file in REQUIRED_FILES:
        path = os.path.join(settings.data_dir, file)
        tablename = os.path.splitext(file)[0]
        if not os.path.isfile(path):
            logger.warning(f"{file} not found, skipping.")
            continue
        try:
            logger.info(f"Loading {file} into DuckDB as '{tablename}'")
            df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
            duckdb_conn.execute(f"CREATE OR REPLACE TABLE {tablename} AS SELECT * FROM df")
        except Exception as e:
            logger.error(f"Failed to load {file}: {e}")
    logger.info("DuckDB tables loaded.")

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

PREBUILT_SYNONYMS = [
    "prebuilt character", "prebuilt characters", "pregens", "sample character", "starter build", 
    "template", "archetype", "statline", "player template", "ready-made", "pregenerated pc", 
    "canon character", "npc template"
]

def is_prebuilt_synonym(query: str) -> bool:
    q = (query or "").lower().strip()
    return any(s in q for s in PREBUILT_SYNONYMS)

def extract_role_gender(query: str, df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    roles = [r.lower().strip() for r in df["Role"].dropna().unique()] if "Role" in df else []
    genders = [g.lower().strip() for g in df["Gender"].dropna().unique()] if "Gender" in df else []
    terms = set([t.lower().strip() for t in re.findall(r'\w+', query)])
    found_role = None
    found_gender = None
    for term in terms:
        if (not found_role) and (term in roles):
            found_role = term
        if (not found_gender) and (term in genders):
            found_gender = term
    return found_role, found_gender

# ----------- FASTAPI APP -----------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    load_duckdb_tables()
    logger.info("Loaded core files.")

@app.get("/lookup")
def lookup(query: Optional[str] = None, file: Optional[str] = None):
    debug = {}
    q = (query or "").strip()
    tablename = "prebuilt_characters"
    debug["query"] = q

    # --- If query is a prebuilt character synonym or empty, return roles/genders/names menu ---
    if is_prebuilt_synonym(q) or not q:
        df = duckdb_conn.execute(f"SELECT * FROM {tablename}").fetchdf()
        roles = sorted(df["Role"].dropna().str.title().unique())
        genders = sorted(df["Gender"].dropna().str.title().unique())
        names = sorted(df["Name"].dropna().unique())
        debug["ambiguous"] = True
        return {
            "debug": debug,
            "clarification_required": True,
            "roles": roles,
            "genders": genders,
            "names": names,
            "message": "Which role and gender? Specify as e.g. 'Solo male', 'Netrunner female'."
        }

    # --- Try to detect role/gender from the query ---
    df = duckdb_conn.execute(f"SELECT * FROM {tablename}").fetchdf()
    role, gender = extract_role_gender(q, df)
    debug["role"] = role
    debug["gender"] = gender

    # --- If both role and gender are detected, search for exact match ---
    if role and gender:
        sql = f"SELECT * FROM {tablename} WHERE lower(Role) = ? AND lower(Gender) = ? LIMIT 10"
        params = [role, gender]
        debug["autodetected_role"] = role
        debug["autodetected_gender"] = gender
        debug["sql"] = sql
        debug["params"] = params
        results = duckdb_conn.execute(sql, params).fetchdf()
        debug["n_results"] = len(results)
        if not results.empty:
            return {
                "debug": debug,
                "rows": results.to_dict(orient="records")
            }
        # If not found, fall through to partial/ambiguous result

    # --- Otherwise, do a broad LIKE match over all text columns (including role/gender/name) ---
    sql = f"""SELECT * FROM {tablename} 
        WHERE (lower(Name) LIKE lower(?) OR lower(Role) LIKE lower(?) OR lower(Gender) LIKE lower(?)) LIMIT 10"""
    params = [f"%{q}%", f"%{q}%", f"%{q}%"]
    debug["sql"] = sql
    debug["params"] = params
    results = duckdb_conn.execute(sql, params).fetchdf()
    debug["n_results"] = len(results)
    if not results.empty:
        return {
            "debug": debug,
            "rows": results.to_dict(orient="records")
        }

    # --- If still nothing, return the menu to prompt clarification ---
    df = duckdb_conn.execute(f"SELECT * FROM {tablename}").fetchdf()
    roles = sorted(df["Role"].dropna().str.title().unique())
    genders = sorted(df["Gender"].dropna().str.title().unique())
    names = sorted(df["Name"].dropna().unique())
    debug["fallback_to_clarification"] = True
    return {
        "debug": debug,
        "clarification_required": True,
        "roles": roles,
        "genders": genders,
        "names": names,
        "message": "No direct match. Specify as e.g. 'Solo male', 'Netrunner female'."
    }

@app.get("/sanity")
def sanity():
    df = duckdb_conn.execute("SELECT * FROM prebuilt_characters").fetchdf()
    return {
        "rows": df[["Name", "Role", "Gender"]].to_dict(orient="records")
    }