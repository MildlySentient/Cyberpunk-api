import os
import logging
import logging.config
import math
import yaml
import re
import string
from typing import List, Dict, Any, Optional, Set, Tuple
import pandas as pd
import duckdb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseSettings
from rapidfuzz import fuzz

from vector_cache import VectorCache

# --- CONFIGURATION ---
VEC_ENABLED = True  # Set to False if you don't want semantic vector fallback

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

# --- PATCH HEADERS FOR _core.tsv FILES ---
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
            if lines[0].strip().lower() != 'file_name':
                logger.info(f"Adding 'File_Name' header to {fname}")
                lines = ['File_Name\n'] + [line if line.endswith('\n') else line + '\n' for line in lines]
                with open(fpath, 'w', encoding='utf-8') as f:
                    f.writelines(lines)

# --- SCHEMA VALIDATION ---
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

# --- DATA FILES TO LOAD ---
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
vector_tables: Dict[str, Tuple[pd.DataFrame, VectorCache]] = {}
keyword_to_file: Dict[str, Set[str]] = {}

# --- Canon role/gender sets for routing ---
canon_roles: Set[str] = set()
canon_genders: Set[str] = set()

# --- UTILITIES ---
def normalize(text):
    return text.strip().lower()

def split_keywords(text):
    """Split on whitespace and punctuation, lower and strip all tokens."""
    tokens = re.split(rf'[\s{re.escape(string.punctuation)}]+', str(text).lower())
    return set(t.strip() for t in tokens if t.strip())

PREBUILT_SYNONYMS = [
    "prebuilt character", "prebuilt characters", "pregens",
    "sample character", "starter build", "template", "archetype",
    "statline", "player template", "ready-made", "pregenerated pc",
    "canon character", "npc template"
]

# --- LOAD FILES (PANDAS + DUCKDB) ---
def load_core_files():
    data_tables.clear()
    vector_tables.clear()
    keyword_to_file.clear()
    canon_roles.clear()
    canon_genders.clear()
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
            # Vectorize prebuilt_characters if enabled
            if VEC_ENABLED and file == "prebuilt_characters.tsv":
                vc = VectorCache()
                vc.preload_vectors(df)
                vector_tables[file] = (df, vc)
                canon_roles.update(r.lower().strip() for r in df["Role"].dropna().unique())
                canon_genders.update(g.lower().strip() for g in df["Gender"].dropna().unique())
        except Exception as e:
            logger.error(f"Failed to load {file}: {e}")
    # Build keyword to file map from Index.tsv
    if "Index.tsv" in data_tables:
        idx = data_tables["Index.tsv"]
        for _, row in idx.iterrows():
            desc = row.get('Description', '')
            fname = row.get('File_Name', '').strip()
            for token in split_keywords(desc):
                keyword_to_file.setdefault(token, set()).add(fname)
            # Also map the lowercased filename as a keyword for explicit matches
            keyword_to_file.setdefault(normalize(fname), set()).add(fname)
    else:
        logger.warning("Index.tsv not loaded; keyword routing will degrade.")

def is_prebuilt_query(query: str) -> bool:
    q = normalize(query)
    idx_df = data_tables.get("Index.tsv", pd.DataFrame())
    desc_tokens = set()
    for _, row in idx_df.iterrows():
        if normalize(row.get("File_Name", "")) == "prebuilt_characters.tsv":
            desc = row.get("Description", "")
            desc_tokens.update(split_keywords(desc))
            break
    # Match direct desc token or synonym
    if any(token in q for token in desc_tokens):
        return True
    return any(x in q for x in PREBUILT_SYNONYMS)

def route_files(query: str) -> List[str]:
    ql = normalize(query)
    words = split_keywords(ql)
    candidates: Set[str] = set()
    # Keyword mapping
    for w in words:
        if w in keyword_to_file:
            candidates.update(keyword_to_file[w])
    # Synonym mapping
    if not candidates and any(x in ql for x in PREBUILT_SYNONYMS):
        candidates.add('prebuilt_characters.tsv')
    # Canon role/gender: critical new logic
    if not candidates and (words & canon_roles or words & canon_genders):
        candidates.add('prebuilt_characters.tsv')
    # Fallback for 'character'/'npc'
    if not candidates and any(x in ql for x in ['character', 'npc']):
        for k, v in keyword_to_file.items():
            if any(x in k for x in ['character', 'npc']):
                candidates.update(v)
        candidates.add('prebuilt_characters.tsv')
    files_found = [f for f in candidates if f in data_tables]
    logger.info(f"route_files: Query='{query}' → files={files_found}, raw_candidates={candidates}")
    return files_found

SYNONYMS = {
    "roll": ["dice", "rolling", "throw", "cast", "d10", "d6", "d100"],
    # Add more synonym sets as needed
}

def extract_role_gender(query: str, df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    roles = [r.lower().strip() for r in df["Role"].dropna().unique()]
    genders = [g.lower().strip() for g in df["Gender"].dropna().unique()]
    terms = set(re.findall(r'\w+', query.lower()))
    role = next((r for r in roles if r in terms), None)
    gender = next((g for g in genders if g in terms), None)
    return role, gender

def match_query(query: str, df: pd.DataFrame, depth: int = 0, tried=None, partials=None) -> Optional[Dict[str, Any]]:
    if tried is None: tried = set()
    if partials is None: partials = []
    ql = normalize(query)
    if ql in tried or depth > 3:
        return None
    tried.add(ql)
    logger.info(f"[Depth {depth}] Matching: {ql}")

    # 1. Direct substring match
    if not df.empty:
        mask = df.apply(lambda r: ql in " ".join(str(x).lower() for x in r), axis=1)
        if mask.any():
            return df[mask].iloc[0].to_dict()
        partials += [str(row.get("Name", "")) for _, row in df[mask].iterrows()]

    # 2. Synonym expansion
    for word, syns in SYNONYMS.items():
        if any(s in ql for s in syns):
            mask = df.apply(lambda r: word in " ".join(str(x).lower() for x in r), axis=1)
            if mask.any():
                return df[mask].iloc[0].to_dict()
            partials += [str(row.get("Name", "")) for _, row in df[mask].iterrows()]

    # 3. Fuzzy partial matching
    for idx, row in df.iterrows():
        try:
            score = fuzz.partial_ratio(ql, " ".join(str(x).lower() for x in row))
            if score > 90:
                return row.to_dict()
            elif score > 70:
                partials.append(row.get("Name", ""))
        except Exception:
            continue

    # 4. Explicit role/gender fallback (for prebuilt)
    if "Role" in df.columns and "Gender" in df.columns:
        role, gender = extract_role_gender(query, df)
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

    # 5. Vector search fallback (semantic)
    if VEC_ENABLED and "Name" in df.columns:
        try:
            file_name = "prebuilt_characters.tsv"  # Only vectorized on this
            if file_name in vector_tables and len(df) == len(vector_tables[file_name][0]):
                vec_cache = vector_tables[file_name][1]
                idx_score_list = vec_cache.find_best_match(query, top_k=1)
                if idx_score_list and idx_score_list[0][1] > 0.7:
                    idx = idx_score_list[0][0]
                    return df.iloc[idx].to_dict()
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")

    return {"message": "No match, tried variants", "variants": partials}

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

# --- FASTAPI APP ---
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
    prebuilt_queried = any(normalize(f) == "prebuilt_characters.tsv" for f in files)
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

# --- SANITY CHECKS FOR PREBUILT CHARACTERS ---
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

@app.get("/_debug-keywords")
def debug_keywords():
    # Inspect what keywords actually map to what files
    return {"keyword_to_file": {k: list(v) for k, v in keyword_to_file.items()}}