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

# ----------- CANONICAL FILE DISCOVERY -----------
def discover_tsv_files(index_path: str, data_dir: str) -> List[str]:
    tsv_files = set()
    tsv_files.add("Index.tsv")
    idx_path = os.path.join(data_dir, index_path)
    if not os.path.isfile(idx_path):
        raise FileNotFoundError(f"index.tsv not found at {idx_path}")
    df = pd.read_csv(idx_path, sep="\t", dtype=str).fillna("")
    for fname in df["File_Name"]:
        if fname.lower().endswith(".tsv"):
            tsv_files.add(fname.strip())
    # Scan each *_core.tsv for children, if present
    for fname in tsv_files.copy():
        if fname.endswith("_core.tsv"):
            fpath = os.path.join(data_dir, fname)
            if os.path.isfile(fpath):
                core_df = pd.read_csv(fpath, sep="\t", dtype=str).fillna("")
                for cf in core_df.get("File_Name", []):
                    if cf and cf.lower().endswith(".tsv"):
                        tsv_files.add(cf.strip())
    return sorted(tsv_files)

REQUIRED_FILES = discover_tsv_files("Index.tsv", settings.data_dir)

# ----------- DUCKDB CANONICAL LOADING -----------
duckdb_conn = duckdb.connect(database=":memory:")

def load_duckdb_tables():
    for file in REQUIRED_FILES:
        path = os.path.join(settings.data_dir, file)
        tablename = os.path.splitext(file)[0].replace("-", "_")
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
    print("DEBUG extract_role_gender:", {"roles": roles, "genders": genders, "terms": list(terms)})
    found_role = None
    found_gender = None
    for term in terms:
        if (not found_role) and (term in roles):
            found_role = term
        if (not found_gender) and (term in genders):
            found_gender = term
    print("  found_role:", found_role, "found_gender:", found_gender)
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
    tables = [t[0] for t in duckdb_conn.execute("SHOW TABLES").fetchall()]
    # Select tablename (if file param provided)
    tablename = None
    if file:
        tablename = os.path.splitext(file)[0].replace("-", "_")
        if tablename not in tables:
            return {"error": f"File '{file}' not loaded or not canonical."}
    # Default to prebuilt_characters if asking about prebuilt/NPCs
    if not tablename and is_prebuilt_synonym(q):
        tablename = "prebuilt_characters"
    # Fallback: search all tables if no file specified
    if not tablename:
        found_rows = []
        for t in tables:
            # Only search canonical data tables, not index or tiny lookup tables
            if t.lower() == "index":
                continue
            try:
                sql = f"""SELECT *, '{t}' as TableName FROM {t} WHERE 
                          (lower(cast({t}.* as varchar)) LIKE ?) LIMIT 5"""
                params = [f"%{q.lower()}%"]
                res = duckdb_conn.execute(sql, params).fetchdf()
                if not res.empty:
                    found_rows.append(res)
            except Exception as e:
                logger.warning(f"Failed broad search in {t}: {e}")
        if found_rows:
            combined = pd.concat(found_rows, ignore_index=True)
            return {"debug": debug, "rows": combined.to_dict(orient="records")}
        # If nothing found, prompt for table/category
        return {"error": f"No canon data found for '{q}'. Please specify a category or try a broader term."}

    # Canonical table search (tablename guaranteed)
    df = duckdb_conn.execute(f"SELECT * FROM {tablename}").fetchdf()
    # Prebuilt/NPC ambiguity menu
    if tablename == "prebuilt_characters" and (is_prebuilt_synonym(q) or not q):
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
    # Try to detect role/gender if searching prebuilt_characters
    if tablename == "prebuilt_characters":
        role, gender = extract_role_gender(q, df)
        debug["role"] = role
        debug["gender"] = gender
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
    # Otherwise, broad LIKE match over all text columns in chosen table
    cols = df.columns
    or_clauses = " OR ".join([f"lower({c}) LIKE ?" for c in cols])
    sql = f"SELECT * FROM {tablename} WHERE {or_clauses} LIMIT 10"
    params = [f"%{q.lower()}%"] * len(cols)
    debug["sql"] = sql
    debug["params"] = params
    results = duckdb_conn.execute(sql, params).fetchdf()
    debug["n_results"] = len(results)
    if not results.empty:
        return {
            "debug": debug,
            "rows": results.to_dict(orient="records")
        }
    # Fallback: ambiguity menu for prebuilt/NPCs
    if tablename == "prebuilt_characters":
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
    return {"error": f"No canon match for '{q}' in '{tablename}'."}

@app.get("/query-canon")
def query_canon(table: str = Query(..., description="TSV file name or canonical table"), 
                field: Optional[str] = None, value: Optional[str] = None):
    safe_table = os.path.splitext(table)[0].replace("-", "_")
    tables = [t[0] for t in duckdb_conn.execute("SHOW TABLES").fetchall()]
    if safe_table not in tables:
        return {"error": f"Table '{safe_table}' not loaded or not canonical."}
    sql = f"SELECT * FROM {safe_table}"
    params = []
    if field and value:
        if not (field in duckdb_conn.execute(f"DESCRIBE {safe_table}").fetchdf()["column_name"].values):
            return {"error": f"Column '{field}' not found in '{safe_table}'."}
        sql += f" WHERE lower({field}) = ?"
        params.append(value.lower())
    results = duckdb_conn.execute(sql, params).fetchdf()
    return {"rows": results.to_dict(orient="records")}

@app.get("/canon-tables")
def canon_tables():
    tables = duckdb_conn.execute("SHOW TABLES").fetchall()
    return {"tables": [t[0] for t in tables]}

@app.get("/sanity")
def sanity():
    try:
        df = duckdb_conn.execute("SELECT * FROM prebuilt_characters").fetchdf()
        return {"rows": df[["Name", "Role", "Gender"]].to_dict(orient="records")}
    except Exception as e:
        return {"error": str(e)}

@app.get("/sanity-dump")
def sanity_dump():
    try:
        df = duckdb_conn.execute("SELECT * FROM prebuilt_characters").fetchdf()
        roles = sorted(df["Role"].dropna().unique())
        genders = sorted(df["Gender"].dropna().unique())
        names = sorted(df["Name"].dropna().unique())
        return {
            "rows": df[["Name", "Role", "Gender"]].to_dict(orient="records"),
            "roles": roles,
            "genders": genders,
            "names": names,
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/sanity-extract")
def sanity_extract(q: str):
    try:
        df = duckdb_conn.execute("SELECT * FROM prebuilt_characters").fetchdf()
        role, gender = extract_role_gender(q, df)
        return {
            "query": q,
            "extracted_role": role,
            "extracted_gender": gender,
            "roles": sorted(df["Role"].dropna().unique()),
            "genders": sorted(df["Gender"].dropna().unique()),
        }
    except Exception as e:
        return {"error": str(e)}