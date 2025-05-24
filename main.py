import os
import logging
import logging.config
import math
import yaml
from typing import List, Dict, Any, Optional, Set, Tuple
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseSettings
from rapidfuzz import fuzz
import re

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

# ----------- DATA CACHE STRUCTURES -----------
data_tables: Dict[str, pd.DataFrame] = {}
vector_tables: Dict[str, Tuple[pd.DataFrame, VectorCache]] = {}
keyword_to_file: Dict[str, Set[str]] = {}

# ----------- DATA LOADING -----------
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

    # Index.tsv loads keyword routing
    if "Index.tsv" in data_tables:
        idx = data_tables["Index.tsv"]
        for _, row in idx.iterrows():
            for word in row['Description'].split(','):
                keyword = word.strip().lower()
                if keyword:
                    keyword_to_file.setdefault(keyword, set()).add(row['File_Name'])
    else:
        logger.warning("Index.tsv not loaded; keyword routing will degrade.")

# ----------- UTILITIES -----------
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

# ----------- DOMAIN LOGIC -----------
PREBUILT_SYNONYMS = [
    "prebuilt character", "prebuilt characters",
    "pregens", "sample character", "sample characters",
    "starter build", "starter builds",
    "template", "templates",
    "archetype", "archetypes",
    "statline", "statlines",
    "player template", "player templates",
    "ready-made", "ready made", "pregenerated pc", "pregenerated pcs",
    "canon character", "canon characters",
    "npc template", "npc templates"
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

# --- Role+Gender Matching & Regex Parsing ---
def extract_role_gender(query: str, df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """Extracts role and gender using regex and normalization."""
    roles = [r.lower().strip() for r in df["Role"].unique()]
    genders = [g.lower().strip() for g in df["Gender"].unique()] if "Gender" in df.columns else []
    ql = query.lower().strip()
    # Regex: find gender then role, or role then gender, or gender in parens
    patterns = [
        r"\b({genders})\b.*?\b({roles})\b".format(genders="|".join(genders), roles="|".join(roles)),
        r"\b({roles})\b.*?\b({genders})\b".format(roles="|".join(roles), genders="|".join(genders)),
        r"\b({roles})\s*\((?P<gender>{genders})\)".format(roles="|".join(roles), genders="|".join(genders)),
        r"\b(?P<gender>{genders})\b".format(genders="|".join(genders)),
        r"\b(?P<role>{roles})\b".format(roles="|".join(roles)),
    ]
    for pat in patterns:
        m = re.search(pat, ql)
        if m:
            # Try to get explicit groups if available
            gender = m.groupdict().get('gender', None)
            role = m.groupdict().get('role', None)
            # If not in named groups, fallback to positional
            if not gender:
                for g in genders:
                    if g in m.groups():
                        gender = g
                        break
            if not role:
                for r in roles:
                    if r in m.groups():
                        role = r
                        break
            if role or gender:
                return (role, gender)
    # Fallback: brute force
    terms = set(ql.split())
    matched_role = next((r for r in roles if r in terms), None)
    matched_gender = next((g for g in genders if g in terms), None)
    return (matched_role, matched_gender)

def match_query(
    query: str,
    df: pd.DataFrame,
    depth: int = 0,
    tried: Optional[Set[str]] = None,
    partials: Optional[List[str]] = None
) -> Optional[Dict[str, Any]]:
    if tried is None:
        tried = set()
    if partials is None:
        partials = []
    ql = query.lower()
    if ql in tried or depth > 3:
        return None
    tried.add(ql)
    logger.info(f"[Depth {depth}] Matching: {ql}")

    # 1) Direct substring match
    if not df.empty:
        mask = df.apply(lambda r: ql in " ".join(str(x).lower() for x in r), axis=1)
        if mask.any():
            return df[mask].iloc[0].to_dict()
        partials += [str(row.get("Name", "")) for _, row in df[mask].iterrows()]

    # 2) Synonym match
    for word, syns in SYNONYMS.items():
        if any(s in ql for s in syns):
            mask = df.apply(lambda r: word in " ".join(str(x).lower() for x in r), axis=1)
            if mask.any():
                return df[mask].iloc[0].to_dict()
            partials += [str(row.get("Name", "")) for _, row in df[mask].iterrows()]

    # 3) Fuzzy match
    for idx, row in df.iterrows():
        try:
            score = fuzz.partial_ratio(ql, " ".join(str(x).lower() for x in row))
            if score > 90:
                return row.to_dict()
            elif score > 70:
                partials.append(row.get("Name", ""))
        except Exception:
            continue

    # 4) Role+Gender fallback for prebuilt_characters.tsv, now with Regex!
    if "Role" in df.columns and "Gender" in df.columns:
        role, gender = extract_role_gender(query, df)
        if role and gender:
            mask = (
                df["Role"].str.lower().str.strip() == role
            ) & (
                df["Gender"].str.lower().str.strip() == gender
            )
            filtered = df[mask]
            if not filtered.empty:
                logger.info(f"Role+Gender fallback matched: Role={role}, Gender={gender}")
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
    load_core_files()
    logger.info("Loaded minimal core files.")

@app.get("/lookup")
def lookup(query: str, file: Optional[str] = None):
    files = [file] if file else route_files(query)
    prebuilt_queried = any(f == "prebuilt_characters.tsv" for f in files)
    responded = False
    for f in files:
        if f not in data_tables:
            continue
        df = data_tables[f]
        result = match_query(query, df)

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
        responded = True

    if prebuilt_queried and "prebuilt_characters.tsv" in data_tables:
        df = data_tables["prebuilt_characters.tsv"]
        roles = sorted(df["Role"].dropna().str.title().unique().tolist()) if "Role" in df else []
        genders = sorted(df["Gender"].dropna().str.title().unique().tolist()) if "Gender" in df else ["Male", "Female"]
        if roles:
            return {
                "code": "clarification_required",
                "message": (
                    "Which role and gender do you want for the prebuilt character? "
                    "Specify as e.g. 'Solo male', 'Netrunner female', etc."
                ),
                "available_roles": roles,
                "available_genders": genders,
            }

    return {
        "code": "not_found",
        "message": "No canonical match. Consult index.tsv."
    }

@app.get("/healthz")
def health():
    return {"status": "ok"}

@app.post("/reload")
def reload_data():
    load_core_files()
    return {"status": "reloaded"}

@app.get("/vector-manifest")
def vector_manifest():
    return {"files": list(vector_tables.keys())}

@app.get("/match")
def match_character(
    query: str,
    use_semantics: bool = True,
    top_k: int = 3,
    score_threshold: float = 0.65,
    role: Optional[str] = None,
    gender: Optional[str] = None
):
    fname = "prebuilt_characters.tsv"
    if fname not in vector_tables:
        raise HTTPException(status_code=503, detail="Vector cache not initialized.")
    df, vc = vector_tables[fname]
    raw_results = vc.find_best_match(query, top_k=top_k * 2)
    matches: List[Dict[str, Any]] = []
    for idx, score in raw_results:
        if score < score_threshold:
            continue
        row = df.iloc[idx].to_dict()
        if role and role.lower() not in str(row.get("Role", "")).lower():
            continue
        if gender and gender.lower() not in str(row.get("Gender", "")).lower():
            continue
        row["_source_file"] = fname
        row["_score"] = round(score, 4)
        matches.append(row)
    seen: Set[str] = set()
    final: List[Dict[str, Any]] = []
    for m in matches:
        key = m.get("Name", "") + m.get("Role", "")
        if key not in seen:
            final.append(m)
            seen.add(key)
        if len(final) >= top_k:
            break
    if not final:
        raise HTTPException(status_code=404, detail="No suitable match found.")
    return final if top_k > 1 else final[0]

@app.post("/vector-reload")
def reload_vector_file(file: str):
    if file != "prebuilt_characters.tsv":
        raise HTTPException(status_code=400, detail="Only prebuilt_characters.tsv is vectorized.")
    path = os.path.join(settings.data_dir, file)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"TSV file '{file}' not found.")
    try:
        df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
        df = prepare_for_matching(df, file)
        vc = VectorCache()
        vc.preload_vectors(df)
        vector_tables[file] = (df, vc)
        return {"status": "reloaded", "file": file, "rows": df.shape[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload vectors for '{file}': {e}")

@app.get("/bootloader")
def show_bootloader():
    path = os.path.join(settings.data_dir, "Bootloader.md")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Bootloader.md not found")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return {"filename": "Bootloader.md", "markdown": content}