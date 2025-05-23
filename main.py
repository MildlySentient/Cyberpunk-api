import os
import glob
import logging
import logging.config
import math
import yaml
from typing import List, Dict, Any, Optional, Set
import pandas as pd
import spacy
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseSettings
from rapidfuzz import fuzz

# ----------- CONFIGURATION -----------

class Settings(BaseSettings):
    cors_origins: List[str] = ["*"]
    data_dir: str = os.path.dirname(__file__)
    logging_config: str = os.path.join(os.path.dirname(__file__), 'logging.yaml')
    spacy_model: str = "en_core_web_md"
    class Config:
        env_file = ".env"

settings = Settings()

# ----------- LOGGING SETUP -----------

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

# ----------- NLP MODEL LOADING -----------

try:
    nlp = spacy.load(settings.spacy_model)
    logger.info(f"Loaded spaCy model: {settings.spacy_model}")
except Exception as e:
    logger.error(f"Failed to load spaCy model: {e}", exc_info=True)
    nlp = None

# ----------- DATA CACHING AND LOADING -----------

class DataCache:
    def __init__(self):
        self.canon_map: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        self.tsv_files: List[str] = []

    def load_tsv_files(self, data_dir: str):
        tsv_pattern = os.path.join(data_dir, '*.tsv')
        self.tsv_files = glob.glob(tsv_pattern)
        logger.info(f"Scanning for .tsv files in: {data_dir}")
        logger.info(f"Found {len(self.tsv_files)} TSV files in canon.")
        self.canon_map = {}
        for tsv in self.tsv_files:
            logger.info(f"Attempting to read {os.path.basename(tsv)}")
            try:
                df = pd.read_csv(tsv, sep='\t', dtype=str).fillna("")
                if "Role" in df.columns and "Gender" in df.columns:
                    for _, row in df.iterrows():
                        role = row["Role"].strip().lower()
                        gender = row["Gender"].strip().lower()
                        if not role or not gender:
                            continue
                        self.canon_map.setdefault(role, {}).setdefault(gender, []).append(row.to_dict())
                logger.info(f"[OK] {os.path.basename(tsv)} – {df.shape[0]} rows")
            except Exception as e:
                logger.error(f"[ERROR] {os.path.basename(tsv)} – {e}", exc_info=True)

cache = DataCache()
cache.load_tsv_files(settings.data_dir)

# ----------- INDEX-BASED ROUTING LOGIC -----------

index_path = os.path.join(settings.data_dir, "Index.tsv")
if os.path.isfile(index_path):
    index_df = pd.read_csv(index_path, sep="\t", dtype=str).fillna("")
    keyword_to_file = {}
    for _, row in index_df.iterrows():
        keywords = [w.strip().lower() for w in row['Description'].split(',')]
        for kw in keywords:
            if kw:
                keyword_to_file.setdefault(kw, set()).add(row['File_Name'])
else:
    logger.warning("Index.tsv not found; keyword routing will not work.")
    keyword_to_file = {}

def route_query_to_files(user_query: str) -> List[str]:
    words = set(user_query.lower().split())
    candidate_files = set()
    for word in words:
        if word in keyword_to_file:
            candidate_files.update(keyword_to_file[word])
    return list(candidate_files)

# ----------- SYNONYMS -----------

SYNONYMS = {
    "roll": ["dice", "rolls", "rolling", "throw", "toss", "cast", "drop", "flip", "chuck", "shake", "d10", "d6", "d100", "percentile", "random", "test", "check", "try my luck", "luck roll", "death roll", "combat roll"],
    "d10": ["ten-sided", "1d10", "d 10", "ten die", "roll d10", "nat 10", "natural 10", "critical roll"],
    "d6": ["six-sided", "1d6", "d 6", "six die", "roll d6"],
    "d100": ["percentile", "1d100", "hundred-sided", "d 100", "roll d100", "percent roll"],
    "initiative": ["combat order", "who goes first", "turn order", "init", "move order", "who acts first"],
}

# ----------- UTILITY FUNCTIONS -----------

def sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        return 0.0 if not math.isfinite(obj) else obj
    elif pd.isna(obj):
        return ""
    else:
        return obj

# ----------- PREBUILT CHARACTER HANDLER -----------

def find_prebuilt_character(query: str, canon_map: Dict[str, Dict[str, list]]) -> Optional[Dict[str, Any]]:
    query_lower = query.lower()
    if "prebuilt character" in query_lower:
        parts = query_lower.replace("prebuilt character", "").strip().split()
        role = next((t for t in parts if t in canon_map), None)
        gender = next((t for t in parts if t in ["male", "female"]), None)
        if role and gender and role in canon_map and gender in canon_map[role]:
            return canon_map[role][gender][0]
        available = []
        for role in canon_map:
            for gender in canon_map[role]:
                available.append(f"{role.title()} {gender.title()}")
        return {
            "message": "Specify a role/gender (e.g., 'Solo male'). Available prebuilt characters: " +
                       ", ".join(available)
        }
    return None

# ----------- RECURSIVE QUERY OVERKILL -----------

def match_query_recursive(
    query: str,
    df: Optional[pd.DataFrame],
    canon_map: Dict[str, Dict[str, list]],
    nlp_model,
    depth=0,
    tried: Optional[Set[str]] = None,
    max_depth=3,
    partials: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    A total, maximum effort matcher. Tries direct, synonym, fuzzy, vector, role/gender, and recursively retries up to 3 layers.
    Returns result or best-effort "close matches" as last resort.
    """
    if tried is None:
        tried = set()
    if partials is None:
        partials = []
    if query in tried or depth > max_depth:
        return None
    tried.add(query)
    logger.info(f"[Depth {depth}] Attempting to match query: {query!r}")

    query_lower = query.lower()

    # 1. Direct string match
    if df is not None and not df.empty:
        mask = df.apply(lambda row: query_lower in " ".join(str(x).lower() for x in row), axis=1)
        if mask.any():
            logger.info(f"[Depth {depth}] Direct match found")
            return df[mask].iloc[0].to_dict()
        # Collect partials for recursion
        partials += [str(row["Name"]) for _, row in df[mask].iterrows() if "Name" in row]

    # 2. Synonym match
    for word, syns in SYNONYMS.items():
        if any(s in query_lower for s in syns):
            if df is not None and not df.empty:
                mask = df.apply(lambda row: word in " ".join(str(x).lower() for x in row), axis=1)
                if mask.any():
                    logger.info(f"[Depth {depth}] Synonym match found for '{word}'")
                    return df[mask].iloc[0].to_dict()
                partials += [str(row["Name"]) for _, row in df[mask].iterrows() if "Name" in row]

    # 3. Fuzzy match (>90)
    if df is not None and not df.empty:
        for idx, row in df.iterrows():
            try:
                score = fuzz.partial_ratio(query_lower, " ".join(str(x).lower() for x in row))
                if score > 90:
                    logger.info(f"[Depth {depth}] Fuzzy match found: score={score}")
                    return row.to_dict()
                elif score > 70:
                    # Only use higher partials, not every single low score
                    partials.append(str(row.get("Name", "")))
            except Exception as e:
                logger.error(f"Fuzzy match error at row {idx}: {e}")

    # 4. Vector similarity (>0.92)
    if nlp_model and df is not None and not df.empty:
        try:
            query_doc = nlp_model(query_lower)
            row_docs = [nlp_model(" ".join(str(x).lower() for x in row)) for _, row in df.iterrows()]
            sims = [query_doc.similarity(row_doc) for row_doc in row_docs]
            if sims and max(sims) > 0.92:
                best_idx = sims.index(max(sims))
                logger.info(f"[Depth {depth}] Vector similarity match found")
                return df.iloc[best_idx].to_dict()
        except Exception as e:
            logger.error(f"Vector similarity error: {e}")

    # 5. Role+Gender fallback
    terms = query_lower.split()
    role = next((t for t in terms if t in canon_map), None)
    gender = next((t for t in terms if t in ["male", "female"]), None)
    if role and gender:
        try:
            logger.info(f"[Depth {depth}] Role+Gender fallback used")
            return canon_map[role][gender][0]
        except Exception:
            pass

    # 6. Try partials recursively (the "madman" fallback)
    if depth < max_depth:
        # Remove empties/duplicates
        next_partials = [p for p in set(partials) if p and p not in tried]
        for part in next_partials:
            logger.info(f"[Depth {depth}] Recursively matching on partial: {part!r}")
            result = match_query_recursive(part, df, canon_map, nlp_model, depth+1, tried, max_depth)
            if result:
                return result

    if depth == 0:
        logger.info("Total query exhaustion: returning all partials as fallback.")
        return {"close_matches": list(set(partials)), "message": "No canonical match, these partials were tried."}

    return None

# ----------- FASTAPI APP SETUP -----------

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_cache():
    return cache

def get_nlp():
    return nlp

@app.get("/canon-map-keys", response_model=Dict[str, List[str]])
def get_canon_map_keys(cache: DataCache = Depends(get_cache)):
    return {"roles": list(cache.canon_map.keys())}

@app.get("/lookup")
def lookup(
    query: str = Query(..., description="Search query"),
    file: str = Query(None, description="TSV filename (optional)"),
    cache: DataCache = Depends(get_cache),
    nlp_model = Depends(get_nlp)
):
    # Special prebuilt handler
    result = find_prebuilt_character(query, cache.canon_map)
    if result:
        return sanitize_for_json(result)

    files_to_search = [file] if file else route_query_to_files(query)
    if not files_to_search:
        logger.error("Could not find relevant data file for query.")
        raise HTTPException(status_code=404, detail="No relevant file found for query.")

    for file_path in files_to_search:
        path = os.path.join(settings.data_dir, file_path)
        if not os.path.isfile(path):
            logger.warning(f"File not found: {path}")
            continue
        try:
            df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
            result = match_query_recursive(query, df, cache.canon_map, nlp_model)
            if result:
                return sanitize_for_json(result)
        except Exception as e:
            logger.error(f"Failed to read file '{path}': {e}", exc_info=True)
            continue

    raise HTTPException(status_code=404, detail="No matching record found in any relevant file.")

@app.get("/healthz")
def health_check():
    return {"status": "ok"}

@app.post("/reload")
def reload_data(cache: DataCache = Depends(get_cache)):
    cache.load_tsv_files(settings.data_dir)
    return {"status": "reloaded"}

if __name__ == "__main__":
    cache.load_tsv_files(settings.data_dir)