import os
import logging
import logging.config
import math
import yaml
import re
from typing import List, Dict, Any, Optional, Set, Tuple
import pandas as pd
import duckdb
from fastapi import FastAPI, Query
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

# ----------- SCHEMA VALIDATION -----------
EXPECTED_SCHEMAS = {
    "prebuilt_characters": ["Name", "Role", "Gender"],
    "index": ["File_Name", "Description", "Category", "Status", "Parent_Core"],
    # Add others as needed
}

def validate_schema(tablename: str, df: pd.DataFrame):
    expected = EXPECTED_SCHEMAS.get(tablename, [])
    found = list(df.columns)
    missing = [col for col in expected if col not in found]
    extra = [col for col in found if col not in expected]
    if missing or extra:
        logger.warning(f"Schema mismatch in '{tablename}': missing {missing}, extra {extra}")

# ----------- CANONICAL FILES (Minimal Preload) -----------
CORE_SYSTEM_PY = ["combat_tracker.py", "combat_resolution.py"]

def get_canonical_files(index_path: str, data_dir: str) -> Dict[str, str]:
    """Return mapping of 'core' file keys to file paths."""
    idx_path = os.path.join(data_dir, index_path)
    files = {}
    if not os.path.isfile(idx_path):
        raise FileNotFoundError(f"Index.tsv not found at {idx_path}")
    df = pd.read_csv(idx_path, sep="\t", dtype=str).fillna("")
    for row in df.to_dict(orient="records"):
        fname = row["File_Name"].strip()
        key = os.path.splitext(fname)[0].replace("-", "_")
        if fname.lower().endswith(".py") and fname in CORE_SYSTEM_PY:
            files[key] = os.path.join(data_dir, fname)
        elif fname.lower() == "prebuilt_characters.tsv":
            files[key] = os.path.join(data_dir, fname)
        elif fname.lower() == "index.tsv":
            files["index"] = os.path.join(data_dir, fname)
    return files

CANONICAL_FILES = get_canonical_files("Index.tsv", settings.data_dir)

# ----------- DUCKDB SETUP (Minimal) -----------
duckdb_conn = duckdb.connect(database=":memory:")

def load_table_to_duckdb(tablename: str, file_path: str):
    try:
        logger.info(f"Loading {file_path} into DuckDB as '{tablename}'")
        if file_path.lower().endswith(".tsv"):
            df = pd.read_csv(file_path, sep="\t", dtype=str).fillna("")
            validate_schema(tablename, df)
            duckdb_conn.execute(f"CREATE OR REPLACE TABLE {tablename} AS SELECT * FROM df")
        elif file_path.lower().endswith(".py"):
            # For .py files, store contents as a single-row table (for API, documentation, or exec)
            with open(file_path, "r") as f:
                code = f.read()
            duckdb_conn.execute(f"CREATE OR REPLACE TABLE {tablename} (filename VARCHAR, code TEXT)")
            duckdb_conn.execute(f"INSERT INTO {tablename} VALUES (?, ?)", (file_path, code))
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")

def preload_minimal_tables():
    for tablename, path in CANONICAL_FILES.items():
        if os.path.isfile(path):
            load_table_to_duckdb(tablename, path)
        else:
            logger.warning(f"{path} not found, skipping preload.")

# ----------- ON-DEMAND TABLE LOADING -----------
def load_table_on_demand(tablename: str) -> Optional[pd.DataFrame]:
    # Always prefer loaded DuckDB table if present
    tables = [t[0] for t in duckdb_conn.execute("SHOW TABLES").fetchall()]
    if tablename in tables:
        try:
            return duckdb_conn.execute(f"SELECT * FROM {tablename}").fetchdf()
        except Exception as e:
            logger.error(f"DuckDB table error for {tablename}: {e}")
    # If not present, try to load from file if listed in index
    idx_path = CANONICAL_FILES.get("index")
    if not idx_path or not os.path.isfile(idx_path):
        logger.error("Index file missing; cannot perform file lookup.")
        return None
    index_df = pd.read_csv(idx_path, sep="\t", dtype=str).fillna("")
    files = {os.path.splitext(f)[0].replace("-", "_"): f for f in index_df["File_Name"]}
    file_name = files.get(tablename)
    if not file_name:
        logger.warning(f"File for '{tablename}' not found in Index.")
        return None
    file_path = os.path.join(settings.data_dir, file_name)
    if not os.path.isfile(file_path):
        logger.warning(f"File {file_path} missing from disk.")
        return None
    # Load and register in DuckDB
    try:
        df = pd.read_csv(file_path, sep="\t", dtype=str).fillna("")
        validate_schema(tablename, df)
        duckdb_conn.execute(f"CREATE OR REPLACE TABLE {tablename} AS SELECT * FROM df")
        return df
    except Exception as e:
        logger.error(f"Failed to load on-demand table {tablename}: {e}")
        return None

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
    preload_minimal_tables()
    logger.info("Loaded canonical files (minimal preload).")

def get_table(tablename: str) -> Optional[pd.DataFrame]:
    tablename = tablename.lower().replace("-", "_")
    tables = [t[0] for t in duckdb_conn.execute("SHOW TABLES").fetchall()]
    if tablename not in tables:
        return load_table_on_demand(tablename)
    try:
        return duckdb_conn.execute(f"SELECT * FROM {tablename}").fetchdf()
    except Exception as e:
        logger.error(f"Failed to access {tablename}: {e}")
        return None

@app.get("/lookup")
def lookup(query: Optional[str] = None, file: Optional[str] = None):
    debug = {}
    q = (query or "").strip().lower()
    tablename = None
    if file:
        tablename = os.path.splitext(file)[0].replace("-", "_").lower()
        df = get_table(tablename)
        if df is None:
            return {"error": f"File '{file}' not loaded or not canonical."}
    else:
        # Default to prebuilt_characters if asking about pregens, archetypes, etc.
        prebuilt_terms = [
            "prebuilt character", "prebuilt characters", "pregens", "sample character", "starter build",
            "template", "archetype", "statline", "player template", "ready-made", "pregenerated pc",
            "canon character", "npc template"
        ]
        if any(term in q for term in prebuilt_terms):
            tablename = "prebuilt_characters"
            df = get_table(tablename)
        else:
            # Try to search all loaded tables (index/prebuilt_characters/core .py files)
            tables = [t[0] for t in duckdb_conn.execute("SHOW TABLES").fetchall()]
            found_rows = []
            for t in tables:
                if t == "index":
                    continue
                try:
                    df = duckdb_conn.execute(f"SELECT * FROM {t}").fetchdf()
                    match = df.apply(lambda row: row.astype(str).str.lower().str.contains(q).any(), axis=1)
                    results = df[match]
                    if not results.empty:
                        results["TableName"] = t
                        found_rows.append(results)
                except Exception as e:
                    logger.warning(f"Search failed in table {t}: {e}")
            if found_rows:
                combined = pd.concat(found_rows, ignore_index=True)
                return {"debug": debug, "rows": combined.to_dict(orient="records")}
            return {"error": f"No canon data found for '{q}'. Please specify a file or broader term."}
    if tablename and df is not None:
        # Default: broad LIKE match over all text columns in chosen table
        results = df[df.apply(lambda row: row.astype(str).str.lower().str.contains(q).any(), axis=1)]
        if not results.empty:
            return {
                "debug": debug,
                "rows": results.to_dict(orient="records")
            }
        return {"error": f"No canon match for '{q}' in '{tablename}'."}
    return {"error": f"No data found for query."}

@app.get("/query-canon")
def query_canon(table: str = Query(..., description="TSV file name or canonical table"),
                field: Optional[str] = None, value: Optional[str] = None):
    safe_table = os.path.splitext(table)[0].replace("-", "_").lower()
    df = get_table(safe_table)
    if df is None:
        return {"error": f"Table '{safe_table}' not loaded or not canonical."}
    if field and value:
        if field not in df.columns:
            return {"error": f"Column '{field}' not found in '{safe_table}'."}
        results = df[df[field].astype(str).str.lower() == value.lower()]
    else:
        results = df
    return {"rows": results.to_dict(orient="records")}

@app.get("/canon-tables")
def canon_tables():
    tables = [t[0] for t in duckdb_conn.execute("SHOW TABLES").fetchall()]
    return {"tables": tables}

@app.get("/sanity")
def sanity():
    df = get_table("prebuilt_characters")
    if df is None:
        return {"error": "prebuilt_characters table not loaded."}
    return {"rows": df[["Name", "Role", "Gender"]].to_dict(orient="records")}

# Optional: Endpoint to see schema warnings (requires in-memory log, otherwise check logs directly)
# (If you want a /schema-warnings endpoint, let me know.)

# End of main.py