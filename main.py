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

# ----------- DATA CACHE -----------

class DataCache:
    def __init__(self):
        self.canon_map: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        self.tsv_files: List[str] = []

    def load_tsv_files(self, data_dir: str):
        tsv_pattern = os.path.join(data_dir, '*.tsv')
        self.tsv_files = glob.glob(tsv_pattern)
        logger.info(f"Found {len(self.tsv_files)} TSV files")
        self.canon_map = {}
        for tsv in self.tsv_files:
            try:
                df = pd.read_csv(tsv, sep='\t', dtype=str).fillna("")
                if "Role" in df.columns and "Gender" in df.columns:
                    for _, row in df.iterrows():
                        role = row["Role"].strip().lower()
                        gender = row["Gender"].strip().lower()
                        if not role or not gender:
                            continue
                        self.canon_map.setdefault(role, {}).setdefault(gender, []).append(row.to_dict())
                logger.info(f"Loaded {os.path.basename(tsv)} ({df.shape[0]} rows)")
            except Exception as e:
                logger.error(f"Failed to read {tsv}: {e}", exc_info=True)

cache = DataCache()
cache.load_tsv_files(settings.data_dir)

# ----------- INDEX LOADING -----------

index_path = os.path.join(settings.data_dir, "Index.tsv")
if os.path.isfile(index_path):
    index_df = pd.read_csv(index_path, sep="\t", dtype=str).fillna("")
    keyword_to_file = {}
    for _, row in index_df.iterrows():
        for word in row['Description'].split(','):
            keyword = word.strip().lower()
            if keyword:
                keyword_to_file.setdefault(keyword, set()).add(row['File_Name'])
else:
    logger.warning("Index.tsv not found; routing will degrade.")
    keyword_to_file = {}

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

SYNONYMS = {
    "roll": ["dice", "rolling", "throw", "cast", "d10", "d6", "d100"],
}
PREBUILT_SYNONYMS = ["prebuilt character", "pregens", "sample character", "starter build"]

def is_prebuilt_query(query: str) -> bool:
    q = query.lower()
    return any(x in q for x in PREBUILT_SYNONYMS)

def route_files(query: str) -> List[str]:
    if is_prebuilt_query(query):
        return ['prebuilt_characters.tsv']
    words = set(query.lower().split())
    candidates = set()
    for w in words:
        candidates.update(keyword_to_file.get(w, set()))
    if not candidates and any(x in query.lower() for x in ['character', 'npc']):
        for k, v in keyword_to_file.items():
            if any(x in k for x in ['character', 'npc']):
                candidates.update(v)
        candidates.add('prebuilt_characters.tsv')
    return list(candidates)

def match_query(query: str, df: pd.DataFrame, depth=0, tried=None, partials=None) -> Optional[Dict[str, Any]]:
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

    if nlp:
        try:
            doc = nlp(ql)
            docs = [nlp(" ".join(str(x).lower() for x in row)) for _, row in df.iterrows()]
            sims = [doc.similarity(d) for d in docs]
            if sims and max(sims) > 0.92:
                return df.iloc[sims.index(max(sims))].to_dict()
        except Exception as e:
            logger.warning(f"NLP error: {e}")

    if depth < 3:
        for p in set(partials) - tried:
            res = match_query(p, df, depth+1, tried)
            if res:
                return res
    
    # Role+Gender fallback
    terms = ql.split()
    role = next((t for t in terms if t in cache.canon_map), None)
    gender = next((t for t in terms if t in ["male", "female"]), None)
    if role and gender:
        try:
            return cache.canon_map[role][gender][0]
        except Exception as e:
            logger.warning(f"Role+Gender fallback failed: {e}")

return {"message": "No match, tried variants", "variants": partials}

# ----------- FASTAPI -----------

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/lookup")
def lookup(query: str, file: Optional[str] = None):
    files = [file] if file else route_files(query)
    for f in files:
        path = os.path.join(settings.data_dir, f)
        if not os.path.isfile(path):
            continue
        try:
            df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
            result = match_query(query, df)
            if result:
                return sanitize(result)
        except Exception as e:
            logger.error(f"Failed to read {f}: {e}", exc_info=True)
    raise HTTPException(status_code=404, detail="No match found")

@app.get("/healthz")
def health():
    return {"status": "ok"}

@app.post("/reload")
def reload_data():
    cache.load_tsv_files(settings.data_dir)
    return {"status": "reloaded"}

@app.get("/canon-map-keys")
def get_canon_map_keys():
    return {"roles": list(cache.canon_map.keys())}
