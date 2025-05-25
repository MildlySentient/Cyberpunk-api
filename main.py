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
from rapidfuzz import fuzz

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

# ----------- PATCH HEADERS FOR _core.tsv FILES -----------
def ensure_core_headers(data_dir, files):
    for fname in files:
        if fname.endswith('_core.tsv'):
            fpath = os.path.join(data_dir, fname)
            if not os.path.isfile(fpath):
                continue
            with open(fpath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            if not lines:
                continue
            # If first line is not 'File_Name', fix it
            if lines[0].strip().lower() != 'file_name':
                logger.info(f"Adding 'File_Name' header to {fname}")
                lines = ['File_Name\n'] + [line if line.endswith('\n') else line + '\n' for line in lines]
                with open(fpath, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
            else:
                logger.debug(f"{fname}: Header already present.")

# ----------- SCHEMA VALIDATION -----------
EXPECTED_SCHEMAS = {
    "prebuilt_characters.tsv": [
        "Name", "Role", "Gender",
        "INT", "REF", "TECH", "COOL", "ATTR", "LUCK", "MA", "BODY", "EMP",
        "Special_Ability", "Special_Ability_Level", "Skills",
        "Lifepath_Cultural_Origin", "Lifepath_Personality", "Lifepath_Family_Background",
        "Lifepath_Motivation", "Lifepath_Friends", "Lifepath_Enemies", "Lifepath_Romance",
        "Gear", "Notes", "Source", "Trigger"
    ],
    "Index.tsv": ["File_Name", "Description", "Category", "Status", "Parent_Core"],
}
for f in [
    "services_core.tsv", "plugins_core.tsv", "campaigns_core.tsv", "corporate_core.tsv", "encounters_core.tsv",
    "setting_core.tsv", "weapons_core.tsv", "assets_core.tsv", "system_core.tsv", "roles_core.tsv",
    "gear_core.tsv", "npc_core.tsv"
]:
    EXPECTED_SCHEMAS[f] = ["File_Name"]

_schema_warnings = []

def validate_schema(filename: str, df: pd.DataFrame):
    expected = EXPECTED_SCHEMAS.get(filename, [])
    found = list(df.columns)
    missing = [col for col in expected if col not in found]
    extra = [col for col in found if col not in expected]
    if filename.endswith("_core.tsv") and expected == ["File_Name"] and (missing == [] and extra == []):
        return
    if missing or extra:
        warning = f"Schema mismatch in '{filename}': missing {missing}, extra {extra}"
        logger.warning(warning)
        _schema_warnings.append(warning)

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

DATA_DIR = settings.data_dir
duckdb_conn = duckdb.connect(database=':memory:')

data_tables: Dict[str, pd.DataFrame] = {}
keyword_to_file: Dict[str, Set[str]] = {}

# ----------- LOAD FILES (PANDAS + DUCKDB) -----------
def load_core_files():
    data_tables.clear()
    keyword_to_file.clear()
    for file in REQUIRED_FILES:
        path = os.path.join(DATA_DIR, file)
        if not os.path.isfile(path):
            logger.warning(f"{file} not found, skipping.")
            continue
        try:
            logger.info(f"Loading {file} into DuckDB and pandas")
            df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
            validate_schema(file, df)
            data_tables[file] = df
            table_name = os.path.splitext(file)[0].replace('.', '_')
            duckdb_conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            duckdb_conn.execute(
                f"CREATE TABLE {table_name} AS SELECT * FROM read_csv_auto('{path}', delim='\t', header=True, IGNORE_ERRORS=TRUE)"
            )
        except Exception as e:
            logger.error(f"Failed to load {file}: {e}")
    # Build keyword to file map from Index.tsv
    if "Index.tsv" in data_tables:
        idx = data_tables["Index.tsv"]
        for _, row in idx.iterrows():
            for word in str(row.get('Description', '')).split(','):
                keyword = word.strip().lower()
                if keyword:
                    keyword_to_file.setdefault(keyword, set()).add(row['File_Name'])
    else:
        logger.warning("Index.tsv not loaded; keyword routing will degrade.")

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
    "prebuilt character", "prebuilt characters", "pregens",
    "sample character", "starter build", "template", "archetype",
    "statline", "player template", "ready-made", "pregenerated pc",
    "canon character", "npc template"
]

def is_prebuilt_query(query: str) -> bool:
    q = query.lower()
    idx_df = data_tables.get("Index.tsv", pd.DataFrame())
    desc_tokens = []
    for _, row in idx_df.iterrows():
        if row.get("File_Name", "").strip().lower() == "prebuilt_characters.tsv":
            desc = row.get("Description", "")
            for token in desc.split(","):
                token = token.strip().lower()
                if token:
                    desc_tokens.append(token)
            break
    if any(token in q for token in desc_tokens):
        return True
    return any(x in q for x in PREBUILT_SYNONYMS)

def route_files(query: str) -> List[str]:
    if is_prebuilt_query(query):
        return ["prebuilt_characters.tsv"]
    words = set(query.lower().split())
    candidates: Set[str] = set()
    for w in words:
        candidates.update(keyword_to_file.get(w, set()))
    if not candidates and any(x in query.lower() for x in ['character', 'npc']):
        for k, v in keyword_to_file.items():
            if any(x in k for x in ['character', 'npc']):
                candidates.update(v)
        candidates.add('prebuilt_characters.tsv')
    return [f for f in candidates if f in data_tables]

def extract_role_gender_any_order(query: str, df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """Order-agnostic matching for role and gender."""
    roles = [r.lower().strip() for r in df["Role"].dropna().unique()]
    genders = [g.lower().strip() for g in df["Gender"].dropna().unique()]
    terms = set(t.lower().strip() for t in re.findall(r'\w+', query))
    found_role = next((t for t in terms if t in roles), None)
    found_gender = next((t for t in terms if t in genders), None)
    return found_role, found_gender

def match_query(query: str, df: pd.DataFrame, depth: int = 0, tried=None, partials=None) -> Optional[Dict[str, Any]]:
    if tried is None: tried = set()
    if partials is None: partials = []
    ql = query.lower()
    if ql in tried or depth > 3:
        return None
    tried.add(ql)
    logger.info(f"[Depth {depth}] Matching: {ql}")

    if not df.empty:
        # Try direct string match
        mask = df.apply(lambda r: ql in " ".join(str(x).lower() for x in r), axis=1)
        if mask.any():
            return df[mask].iloc[0].to_dict()
        partials += [str(row.get("Name", "")) for _, row in df[mask].iterrows()]

    # Try synonym match
    for word, syns in {"roll": ["dice", "rolling", "throw", "cast", "d10", "d6", "d100"]}.items():
        if any(s in ql for s in syns):
            mask = df.apply(lambda r: word in " ".join(str(x).lower() for x in r), axis=1)
            if mask.any():
                return df[mask].iloc[0].to_dict()
            partials += [str(row.get("Name", "")) for _, row in df[mask].iterrows()]

    # Fuzzy
    for idx, row in df.iterrows():
        try:
            score = fuzz.partial_ratio(ql, " ".join(str(x).lower() for x in row))
            if score > 90:
                return row.to_dict()
            elif score > 70:
                partials.append(row.get("Name", ""))
        except Exception:
            continue

    # Order-agnostic role/gender match (key fix)
    if "Role" in df.columns and "Gender" in df.columns:
        role, gender = extract_role_gender_any_order(query, df)
        logger.info(f"[DEBUG] Extracted role={role}, gender={gender} from query='{query}'")
        if role and gender:
            mask = (
                (df["Role"].str.lower().str.strip() == role) &
                (df["Gender"].str.lower().str.strip() == gender)
            )
            filtered = df[mask]
            logger.info(f"[DEBUG] Fallback match count: {filtered.shape[0]}")
            if not filtered.empty:
                return filtered.iloc[0].to_dict()

    return {"message": "No match, tried variants", "variants": partials}

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
    ensure_core_headers(DATA_DIR, REQUIRED_FILES)
    load_core_files()
    logger.info("Loaded core files.")

@app.get("/lookup")
def lookup(query: str, file: Optional[str] = None):
    files = [file] if file else route_files(query)
    prebuilt_queried = any(f == "prebuilt_characters.tsv" for f in files)
    errors = []
    for f in files:
        if f not in data_tables:
            errors.append(f"File {f} not loaded or missing.")
            continue
        df = data_tables[f]
        try:
            result = match_query(query, df)
        except Exception as e:
            errors.append(f"Error searching {f}: {e}")
            continue
        if result and "Name" in result:
            return {
                "source": f,
                "result": sanitize(result),
                "note": f"Returned for query '{query}'"
            }
        if result and "variants" in result:
            roles = sorted(df["Role"].dropna().str.title().unique().tolist()) if "Role" in df else []
            return {
                "code": "ambiguous",
                "message": "Ambiguous query. Please specify role, gender, or consult /canon-map-keys.",
                "available_roles": roles,
            }

    if prebuilt_queried and "prebuilt_characters.tsv" in data_tables:
        df = data_tables["prebuilt_characters.tsv"]
        roles = sorted(df["Role"].dropna().str.title().unique().tolist())
        genders = sorted(df["Gender"].dropna().str.title().unique().tolist())
        return {
            "code": "clarification_required",
            "message": "Which role and gender do you want for the prebuilt character? Specify as e.g. 'Solo male', 'Netrunner female'.",
            "available_roles": roles,
            "available_genders": genders,
        }

    return {
        "code": "not_found",
        "message": "No canonical match. Consult index.tsv.",
        "errors": errors,
    }

# ----------- SANITY CHECKS FOR PREBUILT CHARACTERS -----------
@app.get("/sanity")
def sanity():
    df = data_tables.get("prebuilt_characters.tsv")
    if df is None:
        return {"error": "prebuilt_characters.tsv not loaded."}
    return {"rows": df[["Name", "Role", "Gender"]].to_dict(orient="records")}

@app.get("/sanity-dump")
def sanity_dump():
    df = data_tables.get("prebuilt_characters.tsv")
    if df is None:
        return {"error": "prebuilt_characters.tsv not loaded."}
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
    df = data_tables.get("prebuilt_characters.tsv")
    if df is None:
        return {"error": "prebuilt_characters.tsv not loaded."}
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