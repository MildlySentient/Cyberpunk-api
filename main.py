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
    "prebuilt_characters": [
        "Name", "Role", "Gender",
        "INT", "REF", "TECH", "COOL", "ATTR", "LUCK", "MA", "BODY", "EMP",
        "Special_Ability", "Special_Ability_Level", "Skills",
        "Lifepath_Cultural_Origin", "Lifepath_Personality", "Lifepath_Family_Background",
        "Lifepath_Motivation", "Lifepath_Friends", "Lifepath_Enemies", "Lifepath_Romance",
        "Gear", "Notes", "Source", "Trigger"
    ],
    "index": ["File_Name", "Description", "Category", "Status", "Parent_Core"],
    # Add/expand schemas as needed for other critical files
}

_schema_warnings = []

def validate_schema(tablename: str, df: pd.DataFrame):
    expected = EXPECTED_SCHEMAS.get(tablename, [])
    found = list(df.columns)
    missing = [col for col in expected if col not in found]
    extra = [col for col in found if col not in expected]
    if missing or extra:
        warning = f"Schema mismatch in '{tablename}': missing {missing}, extra {extra}"
        logger.warning(warning)
        _schema_warnings.append(warning)

# ----------- CANONICAL FILES (Minimal Preload) -----------
CANONICAL_FILES = {
    "index": os.path.join(settings.data_dir, "Index.tsv"),
    "prebuilt_characters": os.path.join(settings.data_dir, "prebuilt_characters.tsv"),
    "combat_tracker": os.path.join(settings.data_dir, "combat_tracker.py"),
    "combat_resolution": os.path.join(settings.data_dir, "combat_resolution.py"),
}

duckdb_conn = duckdb.connect(database=":memory:")

def load_table_to_duckdb(tablename: str, file_path: str):
    try:
        logger.info(f"Loading {file_path} into DuckDB as '{tablename}'")
        if file_path.lower().endswith(".tsv"):
            df = pd.read_csv(file_path, sep="\t", dtype=str).fillna("")
            validate_schema(tablename, df)
            duckdb_conn.execute(f"CREATE OR REPLACE TABLE {tablename} AS SELECT * FROM df")
        elif file_path.lower().endswith(".py"):
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
            logger.warning(f"{os.path.basename(path)} not found, skipping preload.")

def get_table(tablename: str) -> Optional[pd.DataFrame]:
    tablename = tablename.lower().replace("-", "_")
    tables = [t[0] for t in duckdb_conn.execute("SHOW TABLES").fetchall()]
    if tablename in tables:
        try:
            return duckdb_conn.execute(f"SELECT * FROM {tablename}").fetchdf()
        except Exception as e:
            logger.error(f"Failed to access {tablename}: {e}")
            return None
    # On-demand load from index if present
    index_path = CANONICAL_FILES.get("index")
    if not index_path or not os.path.isfile(index_path):
        logger.error("Index file missing; cannot perform file lookup.")
        return None
    index_df = pd.read_csv(index_path, sep="\t", dtype=str).fillna("")
    files = {os.path.splitext(f)[0].replace("-", "_").lower(): f for f in index_df["File_Name"]}
    file_name = files.get(tablename)
    if not file_name:
        logger.warning(f"File for '{tablename}' not found in Index.")
        return None
    file_path = os.path.join(settings.data_dir, file_name)
    if not os.path.isfile(file_path):
        logger.warning(f"File {file_path} missing from disk.")
        return None
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

@app.get("/prebuilt")
def get_prebuilt(role: str, gender: str):
    df = get_table("prebuilt_characters")
    if df is None:
        return {"error": "prebuilt_characters table not loaded."}
    role_clean = role.strip().lower()
    gender_clean = gender.strip().lower()
    valid_roles = set(df["Role"].dropna().str.lower())
    valid_genders = set(df["Gender"].dropna().str.lower())
    if role_clean not in valid_roles:
        return {"error": f"Role '{role}' not found in prebuilt_characters."}
    if gender_clean not in valid_genders:
        return {"error": f"Gender '{gender}' not found in prebuilt_characters."}
    results = df[
        (df["Role"].str.lower() == role_clean) &
        (df["Gender"].str.lower() == gender_clean)
    ]
    if results.empty:
        return {"error": f"No prebuilt character found for role '{role}' and gender '{gender}'."}
    return {"character": results.iloc[0].to_dict()}

@app.get("/prebuilt-options")
def prebuilt_options():
    df = get_table("prebuilt_characters")
    if df is None:
        return {"error": "prebuilt_characters table not loaded."}
    roles = sorted(set(df["Role"].dropna()))
    genders = sorted(set(df["Gender"].dropna()))
    return {"roles": roles, "genders": genders}

# ----------- SANITY/DEBUG ENDPOINTS -----------

@app.get("/sanity")
def sanity():
    df = get_table("prebuilt_characters")
    if df is None:
        return {"error": "prebuilt_characters table not loaded."}
    return {"rows": df[["Name", "Role", "Gender"]].to_dict(orient="records")}

@app.get("/sanity-dump")
def sanity_dump():
    df = get_table("prebuilt_characters")
    if df is None:
        return {"error": "prebuilt_characters table not loaded."}
    roles = sorted(df["Role"].dropna().unique())
    genders = sorted(df["Gender"].dropna().unique())
    names = sorted(df["Name"].dropna().unique())
    return {
        "rows": df[["Name", "Role", "Gender"]].to_dict(orient="records"),
        "roles": roles,
        "genders": genders,
        "names": names,
    }

@app.get("/sanity-extract")
def sanity_extract(q: str):
    df = get_table("prebuilt_characters")
    if df is None:
        return {"error": "prebuilt_characters table not loaded."}
    terms = set(t.strip().lower() for t in re.findall(r'\w+', q))
    roles = [r.lower().strip() for r in df["Role"].dropna().unique()]
    genders = [g.lower().strip() for g in df["Gender"].dropna().unique()]
    found_role = next((t for t in terms if t in roles), None)
    found_gender = next((t for t in terms if t in genders), None)
    return {
        "query": q,
        "extracted_role": found_role,
        "extracted_gender": found_gender,
        "roles": sorted(set(roles)),
        "genders": sorted(set(genders)),
    }

@app.get("/schema-warnings")
def schema_warnings():
    return {"schema_warnings": _schema_warnings}