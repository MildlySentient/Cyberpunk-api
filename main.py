import logging
import os
import glob
import json
from typing import List, Union, Optional, Dict, Any, Tuple

import pandas as pd
import spacy
from fastapi import FastAPI, Query, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from rapidfuzz import fuzz
from sklearn.metrics.pairwise import cosine_similarity
import math

# --- Combat modules ---
from cyberpunk2020_engine import roll_d6, roll_d10, roll_d100
from combat_tracker import start_combat, next_turn, get_current_turn
from combat_resolution import resolve_attack

# --- Config ---
MAX_RESULTS = 25
ALLOWED_PAGE_SIZE = 50

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cyberpunk_api")

# --- FastAPI ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Cache ---
cache = {"bootloader": "", "index": None, "cores": {}, "canon_map": {}}

# --- Synonyms for fuzzy matching ---
synonyms = {
    "roll": ["dice", "rolls", "rolling", "throw", "toss", "cast", "drop", "flip", "chuck", "shake", "d10", "d6", "d100", "percentile", "random", "test", "check", "try my luck", "luck roll", "death roll", "combat roll"],
    "d10": ["ten-sided", "1d10", "d 10", "ten die", "roll d10", "nat 10", "natural 10", "critical roll"],
    "d6": ["six-sided", "1d6", "d 6", "six die", "roll d6"],
    "d100": ["percentile", "1d100", "hundred-sided", "d 100", "roll d100", "percent roll"],
    "initiative": ["combat order", "who goes first", "turn order", "init", "move order", "who acts first"],
}

# --- NLP model for lemmatization and vector similarity ---
try:
    nlp = spacy.load("en_core_web_md")
except Exception as e:
    logger.error(f"Failed to load spaCy model: {e}")
    nlp = None

def sanitize_for_json(obj):
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

def load_files():
    tsv_files = glob.glob("**/*.tsv", recursive=True)
    logger.info(f"[INFO] Found {len(tsv_files)} TSV files in canon.")
    cache["canon_map"] = {}
    for tsv in tsv_files:
        try:
            df = pd.read_csv(tsv, sep="\t", dtype=str).fillna("")
            if "Role" in df.columns and "Gender" in df.columns:
                for idx, row in df.iterrows():
                    role = row["Role"].strip().lower()
                    gender = row["Gender"].strip().lower()
                    if not role or not gender:
                        continue
                    cache["canon_map"].setdefault(role, {}).setdefault(gender, []).append((os.path.basename(tsv), row.to_dict()))
            logger.info(f"[OK] {os.path.basename(tsv)} â {df.shape[0]} rows, {df.shape[1]} columns")
        except Exception as e:
            logger.error(f"[ERROR] {os.path.basename(tsv)} â {e}")

@app.get("/canon-map-keys")
def get_canon_map_keys():
    return {"roles": list(cache.get("canon_map", {}).keys())}

def match_query(query: str, df: Optional[pd.DataFrame] = None) -> Optional[Dict[str, Any]]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        logger.warning("No DataFrame provided or DataFrame is empty.")
        return None

    query_lower = query.lower()

    mask = df.apply(lambda row: query_lower in " ".join(str(x).lower() for x in row), axis=1)
    if mask.any():
        return df[mask].iloc[0].to_dict()

    for word, syns in synonyms.items():
        if any(s in query_lower for s in syns):
            mask = df.apply(lambda row: word in " ".join(str(x).lower() for x in row), axis=1)
            if mask.any():
                return df[mask].iloc[0].to_dict()

    for idx, row in df.iterrows():
        try:
            score = fuzz.partial_ratio(query_lower, " ".join(str(x).lower() for x in row))
            if score > 90:
                return row.to_dict()
        except Exception as e:
            logger.error(f"Fuzzy match error at row {idx}: {e}")

    if nlp:
        try:
            query_doc = nlp(query_lower)
            row_docs = [nlp(" ".join(str(x).lower() for x in row)) for _, row in df.iterrows()]
            sims = [query_doc.similarity(row_doc) for row_doc in row_docs]
            if max(sims) > 0.92:
                best_idx = sims.index(max(sims))
                return df.iloc[best_idx].to_dict()
        except Exception as e:
            logger.error(f"Vector similarity error: {e}")

    terms = query_lower.split()
    role = next((t for t in terms if t in cache["canon_map"]), None)
    gender = next((t for t in terms if t in ["male", "female"]), None)
    if role and gender:
        try:
            return cache["canon_map"][role][gender][0][1]
        except Exception as e:
            logger.error(f"Canon map fallback error: {e}")

    logger.info(f"No match for query: '{query}'")
    return None

@app.get("/lookup")
def lookup(query: str, file: str):
    try:
        df = pd.read_csv(file, sep="\t", dtype=str).fillna("")
    except FileNotFoundError as e:
        logger.error(f"File not found: {file} â {e}")
        raise HTTPException(status_code=404, detail=f"File '{file}' not found")
    except Exception as e:
        logger.error(f"Error loading file '{file}': {e}")
        raise HTTPException(status_code=500, detail="File loading error")
    result = match_query(query, df)
    if result:
        return sanitize_for_json(result)
    raise HTTPException(status_code=404, detail="No matching record found.")

# Initial load
load_files()
