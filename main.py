import os
import glob
import logging
import logging.config
import math
import yaml
from typing import List, Dict, Any, Optional, Set, Tuple
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
    keyword_to_file: Dict[str, Set[str]] = {}
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
    candidates: Set[str] = set()
    for w in words:
        candidates.update(keyword_to_file.get(w, set()))
    if not candidates and any(x in query.lower() for x in ['character', 'npc']):
        for k, v in keyword_to_file.items():
            if any(x in k for x in ['character', 'npc']):
                candidates.update(v)
        candidates.add('prebuilt_characters.tsv')
    return list(candidates)

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

    # 4) Vector similarity
    if nlp:
        try:
            doc = nlp(ql)
            docs = [nlp(" ".join(str(x).lower() for x in row)) for _, row in df.iterrows()]
            sims = [doc.similarity(d) for d in docs]
            if sims and max(sims) > 0.92:
                return df.iloc[sims.index(max(sims))].to_dict()
        except Exception as e:
            logger.warning(f"NLP error: {e}")

    # 5) Recursive partials
    if depth < 3:
        for p in set(partials) - tried:
            res = match_query(p, df, depth + 1, tried, partials)
            if res:
                return res

    # 6) Role+Gender fallback
    terms = ql.split()
    role = next((t for t in terms if t in cache.canon_map), None)
    gender = next((t for t in terms if t in ["male", "female"]), None)
    if role and gender:
        try:
            return cache.canon_map[role][gender][0]
        except Exception as e:
            logger.warning(f"Role+Gender fallback failed: {e}")

    # ---- Final return must be *inside* the function ----
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

# ----------- VECTOR CACHE INTEGRATION -----------
from vector_cache import VectorCache

vector_cache = VectorCache()
vector_df: Optional[pd.DataFrame] = None

@app.on_event("startup")
def extended_startup_event():
    global vector_df
    from preprocessing import prepare_for_matching
    tsv_path = os.path.join(settings.data_dir, "prebuilt_characters.tsv")
    if os.path.isfile(tsv_path):
        df = pd.read_csv(tsv_path, sep="\t", dtype=str).fillna("")
        df = prepare_for_matching(df, "prebuilt_characters.tsv")
        vector_df = df
        vector_cache.preload_vectors(df)
    else:
        raise RuntimeError("Required TSV not found for vector search.")

@app.get("/match")
def match_character(query: str, use_semantics: bool = True, top_k: int = 1):
    global vector_df
    if vector_df is None:
        raise HTTPException(status_code=503, detail="Vector cache not initialized.")
    if use_semantics:
        results = vector_cache.find_best_match(query, top_k=top_k)
        matches = [vector_df.iloc[i].to_dict() | {"_score": round(score, 4)} for i, score in results]
        return matches if top_k > 1 else matches[0]
    mask = vector_df["__match_text_internal__"].str.contains(query.lower())
    if mask.any():
        return vector_df[mask].iloc[0].to_dict()
    raise HTTPException(status_code=404, detail="No match found.")

# ----------- EXPANDED VECTOR MATCHING -----------
vector_tables: Dict[str, Tuple[pd.DataFrame, VectorCache]] = {}

def load_vector_sources():
    global vector_tables
    vector_tables.clear()
    for file in os.listdir(settings.data_dir):
        if file.endswith(".tsv"):
            full_path = os.path.join(settings.data_dir, file)
            try:
                df = pd.read_csv(full_path, sep="\t", dtype=str).fillna("")
                from preprocessing import prepare_for_matching
                df = prepare_for_matching(df, file)
                vc = VectorCache()
                vc.preload_vectors(df)
                vector_tables[file] = (df, vc)
            except Exception as e:
                logger.warning(f"Failed to load {file} into vector cache: {e}")

@app.on_event("startup")
def load_all_vectors():
    load_vector_sources()

@app.get("/match")
def match_character(
    query: str,
    use_semantics: bool = True,
    top_k: int = 3,
    score_threshold: float = 0.65,
    role: Optional[str] = None,
    gender: Optional[str] = None
):
    matches: List[Dict[str, Any]] = []
    for fname, (df, vc) in vector_tables.items():
        if use_semantics:
            raw_results = vc.find_best_match(query, top_k=top_k * 2)
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
    # Dedupe by Name+Role
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

@app.get("/vector-manifest")
def vector_manifest():
    return {"files": list(vector_tables.keys())}

@app.post("/vector-reload")
def reload_vector_file(file: str):
    global vector_tables
    from preprocessing import prepare_for_matching
    full_path = os.path.join(settings.data_dir, file)
    if not os.path.isfile(full_path):
        raise HTTPException(status_code=404, detail=f"TSV file '{file}' not found.")
    try:
        df = pd.read_csv(full_path, sep="\t", dtype=str).fillna("")
        df = prepare_for_matching(df, file)
        vc = VectorCache()
        vc.preload_vectors(df)
        vector_tables[file] = (df, vc)
        return {"status": "reloaded", "file": file, "rows": df.shape[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload vectors for '{file}': {e}")