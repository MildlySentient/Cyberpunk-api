import logging.config
import os
import glob
import traceback
from typing import Dict, Any, Optional, List

import pandas as pd
import spacy
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from rapidfuzz import fuzz

# --- Load environment variables (for config) ---
load_dotenv()
DATA_PATH = os.environ.get("DATA_PATH", "data")  # Folder with TSVs

# --- Centralized Logging Setup ---
import yaml
with open(os.path.join(os.path.dirname(__file__), "logging.yaml"), "r") as f:
    logging.config.dictConfig(yaml.safe_load(f))
logger = logging.getLogger("cyberpunk_api")

# --- Synonyms for fuzzy matching ---
SYNONYMS = {
    "roll": ["dice", "rolls", "rolling", "throw", "toss", "cast", "drop", "flip", "chuck", "shake", "d10", "d6", "d100", "percentile", "random", "test", "check", "try my luck", "luck roll", "death roll", "combat roll"],
    "d10": ["ten-sided", "1d10", "d 10", "ten die", "roll d10", "nat 10", "natural 10", "critical roll"],
    "d6": ["six-sided", "1d6", "d 6", "six die", "roll d6"],
    "d100": ["percentile", "1d100", "hundred-sided", "d 100", "roll d100", "percent roll"],
    "initiative": ["combat order", "who goes first", "turn order", "init", "move order", "who acts first"],
}

# --- NLP Model ---
try:
    nlp = spacy.load("en_core_web_md")
except Exception as e:
    logger.error("Failed to load spaCy model: %s\n%s", e, traceback.format_exc())
    nlp = None

# --- Canon Cache and File Index ---
cache = {
    "canon_map": {},
    "tsv_index": [],
}

# --- Pydantic Schemas ---
class CanonicalEntry(BaseModel):
    role: str
    gender: str
    file: str
    data: Dict[str, Any]

class CanonKeys(BaseModel):
    roles: List[str]

class TSVList(BaseModel):
    files: List[str]

class StatusMessage(BaseModel):
    status: str
    file_count: int

# --- Custom Exceptions ---
class CanonicalEntryNotFound(Exception):
    def __init__(self, role: str, gender: str):
        self.role = role
        self.gender = gender
        super().__init__(f"No canonical entry for role '{role}' and gender '{gender}'.")

# --- Exception Handlers ---
def add_exception_handlers(app: FastAPI):
    @app.exception_handler(CanonicalEntryNotFound)
    async def handle_canonical_entry_not_found(request: Request, exc: CanonicalEntryNotFound):
        return JSONResponse(
            status_code=404,
            content={"detail": str(exc)},
        )
    return app

# --- Data Utilities ---
def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        return 0.0 if not pd.notna(obj) or not pd.api.types.is_number(obj) else obj
    elif pd.isna(obj):
        return ""
    else:
        return obj

def load_files() -> None:
    """Loads all .tsv files and builds canon_map and tsv_index."""
    data_dir = os.path.abspath(DATA_PATH)
    tsv_files = glob.glob(os.path.join(data_dir, "**/*.tsv"), recursive=True)
    cache["canon_map"] = {}
    cache["tsv_index"] = []
    logger.info("Found %d TSV files.", len(tsv_files))
    for tsv_path in tsv_files:
        try:
            df = pd.read_csv(tsv_path, sep="\t", dtype=str, keep_default_na=False).fillna("")
            filename = os.path.basename(tsv_path)
            cache["tsv_index"].append(filename)
            if "Role" in df.columns and "Gender" in df.columns:
                for _, row in df.iterrows():
                    role = row.get("Role", "").strip().lower()
                    gender = row.get("Gender", "").strip().lower()
                    if role and gender:
                        cache["canon_map"].setdefault(role, {}).setdefault(gender, []).append((filename, row.to_dict()))
            logger.info("[OK] %-28s | %4d rows x %3d cols", filename, df.shape[0], df.shape[1])
        except Exception as e:
            logger.error("[FAIL] %-28s | %s\n%s", tsv_path, e, traceback.format_exc())

def get_first_canon(role: str, gender: str) -> CanonicalEntry:
    """Return the first canonical entry for a given role/gender."""
    canon = cache["canon_map"].get(role.lower(), {}).get(gender.lower())
    if canon and canon[0]:
        filename, data = canon[0]
        return CanonicalEntry(role=role, gender=gender, file=filename, data=data)
    raise CanonicalEntryNotFound(role, gender)

def match_query(query: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    query_lower = query.strip().lower()
    columns_joined = lambda row: " ".join(str(x).lower() for x in row)
    # Direct match
    mask = df.apply(lambda row: query_lower in columns_joined(row), axis=1)
    if mask.any():
        return df[mask].iloc[0].to_dict()
    # Synonym match
    for word, syns in SYNONYMS.items():
        if any(s in query_lower for s in syns):
            mask = df.apply(lambda row: word in columns_joined(row), axis=1)
            if mask.any():
                return df[mask].iloc[0].to_dict()
    # Fuzzy match
    for idx, row in df.iterrows():
        score = fuzz.partial_ratio(query_lower, columns_joined(row))
        if score > 90:
            return row.to_dict()
    # Vector similarity
    if nlp:
        query_doc = nlp(query_lower)
        row_docs = [nlp(columns_joined(row)) for _, row in df.iterrows()]
        sims = [query_doc.similarity(row_doc) for row_doc in row_docs]
        if sims and max(sims) > 0.92:
            best_idx = sims.index(max(sims))
            return df.iloc[best_idx].to_dict()
    # Canon fallback
    terms = query_lower.split()
    role = next((t for t in terms if t in cache["canon_map"]), None)
    gender = next((t for t in terms if t in ("male", "female")), None)
    if role and gender:
        return get_first_canon(role, gender).data
    return None

# --- FastAPI Lifespan Event (for startup tasks) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_files()
    yield

# --- App Factory ---
def create_app():
    app = FastAPI(
        title="Cyberpunk GPT Canon API",
        description="Serves canonical NPC data and dynamic lookup from TSVs.",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Restrict in prod!
        allow_methods=["*"],
        allow_headers=["*"],
    )
    add_exception_handlers(app)

    @app.get("/canon-map-keys", response_model=CanonKeys)
    def get_canon_map_keys():
        """Get all canonical roles currently indexed."""
        return CanonKeys(roles=list(cache.get("canon_map", {}).keys()))

    @app.get("/tsv-list", response_model=TSVList)
    def get_tsv_list():
        """List all indexed TSV filenames."""
        return TSVList(files=cache.get("tsv_index", []))

    @app.get("/reload", response_model=StatusMessage)
    def reload_files_api():
        """Force reload of all TSV files."""
        load_files()
        return StatusMessage(status="reloaded", file_count=len(cache.get("tsv_index", [])))

    @app.get("/canonical-entry", response_model=CanonicalEntry)
    def canonical_entry(role: str = Query(...), gender: str = Query(...)):
        """Retrieve the first canonical NPC entry for a role/gender pair."""
        return get_first_canon(role, gender)

    @app.get("/lookup")
    def lookup(query: str = Query(...), file: str = Query(...)):
        """Lookup a record in a specific TSV by query (uses direct/synonym/fuzzy/vector/canon)."""
        if not file.endswith(".tsv"):
            raise HTTPException(status_code=400, detail="File must be a .tsv")
        if file not in cache["tsv_index"]:
            raise HTTPException(status_code=404, detail=f"File '{file}' not found in TSV index")
        matches = glob.glob(os.path.join(DATA_PATH, "**", file), recursive=True)
        if not matches:
            raise HTTPException(status_code=404, detail=f"File '{file}' not found on disk")
        tsv_path = matches[0]
        df = pd.read_csv(tsv_path, sep="\t", dtype=str, keep_default_na=False).fillna("")
        result = match_query(query, df)
        if result:
            return sanitize_for_json(result)
        raise HTTPException(status_code=404, detail="No matching record found.")

    return app

# --- Entrypoint for Uvicorn/Gunicorn ---
app = create_app()