import os
import logging
import logging.config
import math
import yaml
import re
from typing import List, Dict, Any, Optional, Set, Tuple
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseSettings
from rapidfuzz import fuzz

from vector_cache import VectorCache

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

data_tables: Dict[str, pd.DataFrame] = {}
vector_tables: Dict[str, Tuple[pd.DataFrame, VectorCache]] = {}
keyword_to_file: Dict[str, Set[str]] = {}

def prepare_for_matching(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    df["__match_text_internal__"] = df.astype(str).apply(" ".join, axis=1).str.lower()
    return df

def load_core_files():
    data_tables.clear()
    vector_tables.clear()
    for file in REQUIRED_FILES:
        path = os.path.join(settings.data_dir, file)
        if not os.path.isfile(path):
            logger.warning(f"{file} not found, skipping.")
            continue
        try:
            df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
            if file == "prebuilt_characters.tsv":
                df = prepare_for_matching(df, file)
                vc = VectorCache()
                vc.preload_vectors(df)
                vector_tables[file] = (df, vc)
            data_tables[file] = df
            logger.info(f"Loaded {file} ({df.shape[0]} rows)")
        except Exception as e:
            logger.error(f"Failed to load {file}: {e}")

    if "Index.tsv" in data_tables:
        idx = data_tables["Index.tsv"]
        for _, row in idx.iterrows():
            for word in row['Description'].split(','):
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

SYNONYMS = {
    "roll": ["dice", "rolling", "throw", "cast", "d10", "d6", "d100"],
}

# ----- PATCHED FUNCTION -----
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
    print(f"DEBUG: extracted role={found_role}, gender={found_gender}")
    return found_role, found_gender
# ----------------------------

def match_query(query: str, df: pd.DataFrame, depth: int = 0, tried=None, partials=None) -> Optional[Dict[str, Any]]:
    if tried is None: tried = set()
    if partials is None: partials = []
    ql = query.lower()
    if ql in tried or depth > 3:
        return None
    tried.add(ql)
    logger.info(f"[Depth {depth}] Matching: {ql}")

    if not df.empty:
        mask = df.apply(lambda r: ql in " ".join(str(x).lower() for x in r), axis=1)
        if mask.any():
            return df[mask].iloc[0].to_dict()
        partials += [str(row.get("Name", "")) for _, row in df[mask].iterrows()]

    for word, syns in SYNONYMS.items():
        if any(s in ql for s in syns):
            mask = df.apply(lambda r: word in " ".join(str(x).lower() for x in r), axis=1)
            if mask.any():
                return df[mask].iloc[0].to_dict()
            partials += [str(row.get("Name", "")) for _, row in df[mask].iterrows()]

    for idx, row in df.iterrows():
        try:
            score = fuzz.partial_ratio(ql, " ".join(str(x).lower() for x in row))
            if score > 90:
                return row.to_dict()
            elif score > 70:
                partials.append(row.get("Name", ""))
        except Exception:
            continue

if "Role" in df.columns and "Gender" in df.columns:
    role, gender = extract_role_gender(query, df)
    logger.info(f"[DEBUG] Extracted role={role}, gender={gender} from query='{query}'")
    print(f"DEBUG: Filtering for role='{role}', gender='{gender}'")
    # Print all roles and genders for every row (sanity check)
    print("All rows:", df[["Name", "Role", "Gender"]].to_dict(orient="records"))
    if role and gender:
        mask = (
            (df["Role"].str.lower().str.strip() == role) &
            (df["Gender"].str.lower().str.strip() == gender)
        )
        filtered = df[mask]
        print("Filtered result:", filtered[["Name", "Role", "Gender"]].to_dict(orient="records"))
        logger.info(f"[DEBUG] Fallback match count: {filtered.shape[0]}")
        if not filtered.empty:
            return filtered.iloc[0].to_dict()

    return {"message": "No match, tried variants", "names": partials}

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    load_core_files()
    logger.info("Loaded core files.")

@app.get("/lookup")
def lookup(query: str, file: Optional[str] = None):
    files = [file] if file else route_files(query)
    prebuilt_queried = any(f == "prebuilt_characters.tsv" for f in files)
    for f in files:
        if f not in data_tables:
            continue
        df = data_tables[f]
        result = match_query(query, df)
        # Canonical match: has Name field, return directly
        if result and "Name" in result:
            return {
                "source": f,
                "result": sanitize(result),
                "note": f"Returned for query '{query}'"
            }
        # Ambiguous result: return with roles/genders/names arrays
        if result and "names" in result:
            roles = sorted(df["Role"].dropna().str.title().unique().tolist()) if "Role" in df else []
            genders = sorted(df["Gender"].dropna().str.title().unique().tolist()) if "Gender" in df else []
            names = result.get("names", [])
            return {
                "code": "ambiguous",
                "message": "Ambiguous query. Please specify role, gender, or consult /canon-map-keys.",
                "roles": roles,
                "genders": genders,
                "names": names
            }

    # Prebuilt clarification (force roles/genders/names)
    if prebuilt_queried and "prebuilt_characters.tsv" in data_tables:
        df = data_tables["prebuilt_characters.tsv"]
        roles = sorted(df["Role"].dropna().str.title().unique().tolist()) if "Role" in df else []
        genders = sorted(df["Gender"].dropna().str.title().unique().tolist()) if "Gender" in df else []
        names = sorted(df["Name"].dropna().unique().tolist()) if "Name" in df else []
        return {
            "code": "clarification_required",
            "message": "Which role and gender do you want for the prebuilt character? Specify as e.g. 'Solo male', 'Netrunner female'.",
            "roles": roles,
            "genders": genders,
            "names": names
        }

    # Not found: always return arrays
    return {
        "code": "not_found",
        "message": "No canonical match. Consult index.tsv.",
        "roles": [],
        "genders": [],
        "names": []
    }
@app.get("/sanity")
def sanity():
    df = data_tables.get("prebuilt_characters.tsv")
    if df is None:
        return {"error": "No data"}
    # Print all roles/genders combos
    return {
        "rows": df[["Name", "Role", "Gender"]].to_dict(orient="records")
    }