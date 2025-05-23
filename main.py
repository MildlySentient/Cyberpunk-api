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
cache = {"bootloader": "", "index": None, "cores": {}}

# --- Synonyms for fuzzy matching ---
synonyms = {
    "roll": ["dice", "rolls", "rolling", "throw", "toss", "cast", "drop", "flip", "chuck", "shake", "d10", "d6", "d100", "percentile", "random", "test", "check", "try my luck", "luck roll", "death roll", "combat roll"],
    "d10": ["ten-sided", "1d10", "d 10", "ten die", "roll d10", "nat 10", "natural 10", "critical roll"],
    "d6": ["six-sided", "1d6", "d 6", "six die", "roll d6"],
    "d100": ["percentile", "1d100", "hundred-sided", "d 100", "roll d100", "percent roll"],
    "initiative": ["combat order", "who goes first", "turn order", "init", "move order", "who acts first", "combat initiative"],
    "hack": ["netrun", "breach", "jack in", "netrunning", "run the net", "crack", "intrude", "exploit", "penetrate", "bypass", "ICE breaker", "cyberjack", "data tap", "deck", "backdoor", "console cowboy", "run a hack"],
    "netrunner": ["hacker", "net jockey", "decker", "cyberjack", "runner", "console cowboy", "sysop", "black hat", "white hat", "cowboy", "cracker", "data thief", "script kiddie", "netrat", "netscape"],
    "cyberware": ["cyberwear", "chrome", "implant", "ware", "cybertech", "augmentation", "aug", "cybermod", "cybernetics", "hardware", "wetware", "nanoware", "neuralware", "mod", "modded", "chrome up", "socket", "cyber up", "plug-in"],
    # ... add many more for all core terms and slang ...
}

# --- spaCy model ---
try:
    nlp = spacy.load("en_core_web_md")
except Exception as e:
    logger.warning(f"spaCy model load error: {e}")
    nlp = None

def lemmatize(text):
    if not nlp or pd.isna(text):
        return str(text) if text is not None else ""
    return " ".join(token.lemma_ for token in nlp(str(text)))

def load_files():
    data_folder = os.path.dirname(__file__)
    try:
        with open(os.path.join(data_folder, "Bootloader.md"), "r", encoding="utf-8") as f:
            cache["bootloader"] = f.read()
    except Exception:
        cache["bootloader"] = "[ERROR] Bootloader.md not found."
    try:
        index_path = os.path.join(data_folder, "index.tsv")
        index = pd.read_csv(index_path, sep="\t")
        index["Description"] = index["Description"].fillna("").astype(str)
        index["LemDescription"] = index["Description"].apply(lemmatize)
        cache["index"] = index
    except Exception:
        cache["index"] = pd.DataFrame([{"Filename": "", "Description": "", "LemDescription": ""}])
    cache["cores"] = {}

    cache["canon_map"] = {}
    for file in os.listdir(data_folder):
        if file.endswith("_core.tsv"):
            try:
                df = pd.read_csv(os.path.join(data_folder, file), sep="\t")
                if "Role" in df.columns:
                    for _, row in df.iterrows():
                        role = str(row.get("Role", "")).lower()
                        gender = str(row.get("Gender", "")).lower()
                        if role:
                            cache["canon_map"].setdefault(role, {}).setdefault(gender, []).append((file, row.to_dict()))
            except Exception:
                continue

    for file in os.listdir(data_folder):
        if file.endswith("_core.tsv"):
            try:
                df = pd.read_csv(os.path.join(data_folder, file), sep="\t")
                df["Trigger"] = df.get("Trigger", "").fillna("").astype(str)
                df["LemTrigger"] = df["Trigger"].apply(lemmatize)
                cache["cores"][file] = df
            except Exception:
                pass

def match_query(query, col, df):
    if df is None or df.empty or col not in df.columns:
        return {}
    query_lem = lemmatize(query)

    # Synonym matching
    for base, syns in synonyms.items():
        if any(term in query_lem for term in [base] + syns):
            mask = df[col].str.contains(base, na=False, case=False)
            match = df[mask]
            if not match.empty:
                return match.iloc[0].to_dict()

    # Fuzzy match
    fuzz_scores = df[col].apply(lambda x: fuzz.partial_ratio(query_lem, str(x)))
    if fuzz_scores.max() > 85:
        return df.iloc[fuzz_scores.idxmax()].to_dict()

    # Vector similarity
    if nlp:
        q_vec = nlp(query_lem).vector.reshape(1, -1)
        best = None
        best_score = -1
        for idx, row in df.iterrows():
            text = row[col]
            if not isinstance(text, str) or not text.strip():
                continue
            row_vec = nlp(text).vector.reshape(1, -1)
            sim = cosine_similarity(q_vec, row_vec)[0][0]
            if sim > best_score:
                best_score = sim
                best = row.to_dict()
        if best:
            return best

    # Role + Gender fallback
    if "Role" in df.columns and "Gender" in df.columns:
        terms = query.lower().split()
        possible_role = next((term.capitalize() for term in terms if term.capitalize() in df["Role"].unique()), None)
        possible_gender = next((g.capitalize() for g in terms if g.lower() in ["male", "female"]), None)
        if possible_role and possible_gender:
            role_match = df[(df["Role"] == possible_role) & (df["Gender"] == possible_gender)]
            if not role_match.empty:
                return role_match.iloc[0].to_dict()
    return {}
    if df is None or df.empty or col not in df.columns:
        return {}
    query_lem = lemmatize(query)
    for base, syns in synonyms.items():
        if any(term in query_lem for term in [base] + syns):
            mask = df[col].str.contains(base, na=False, case=False)
            match = df[mask]
            if not match.empty:
                return match.iloc[0].to_dict()
    fuzz_scores = df[col].apply(lambda x: fuzz.partial_ratio(query_lem, str(x)))
    if fuzz_scores.max() > 85:
        return df.iloc[fuzz_scores.idxmax()].to_dict()
    if nlp:
        q_vec = nlp(query_lem).vector.reshape(1, -1)
        best = None
        best_score = -1
        for idx, row in df.iterrows():
            text = row[col]
            if not isinstance(text, str) or not text.strip():
                continue
            row_vec = nlp(text).vector.reshape(1, -1)
            sim = cosine_similarity(q_vec, row_vec)[0][0]
            if sim > best_score:
                best_score = sim
                best = row.to_dict()
        if best:
            return best
    return {}

def fallback_file_lookup(query, data_folder):
    # Find any .tsv in the data folder
    all_tsvs = [os.path.basename(p) for p in glob.glob(os.path.join(data_folder, "*.tsv"))]
    candidates = [(f, fuzz.partial_ratio(query.lower(), f.lower())) for f in all_tsvs]
    candidates.sort(key=lambda x: x[1], reverse=True)
    # Only strong matches, e.g., >60
    for fname, score in candidates:
        if score > 60:
            try:
                df = pd.read_csv(os.path.join(data_folder, fname), sep="\t")
                # Optional: Search inside file for the query as well
                for col in df.columns:
                    if df[col].astype(str).str.contains(query, case=False, na=False).any():
                        return fname, df
            except Exception:
                continue
    # If still nothing, just return top filename as a last resort
    if candidates:
        try:
            df = pd.read_csv(os.path.join(data_folder, candidates[0][0]), sep="\t")
            return candidates[0][0], df
        except Exception:
            pass
    return None, None

# --- Models (same as before) ---
class QueryRequest(BaseModel):
    query: str = Field(..., description="Text query for game data")
    page: Optional[int] = Field(1, description="Page number for pagination")
    page_size: Optional[int] = Field(MAX_RESULTS, description="Results per page")
    fields: Optional[str] = Field(None, description="Comma-separated fields to return")

class CombatantsModel(BaseModel):
    combatants: List[str]

class CombatResolveModel(BaseModel):
    attacker: str
    target: str
    weapon: str
    armor: str

class SaveGameRequest(BaseModel):
    game_id: str
    state: dict

class LoadGameRequest(BaseModel):
    game_id: str

class DataResultModel(BaseModel):
    source: str
    result: Union[str, Dict[str, Any], List[Dict[str, Any]]]
    note: Optional[str] = None

class DiceResultModel(BaseModel):
    result: Optional[int] = None
    error: Optional[str] = None

class CombatResultModel(BaseModel):
    result: Optional[Any] = None
    error: Optional[str] = None

SAVE_DIR = os.path.join(os.path.dirname(__file__), "saves")
os.makedirs(SAVE_DIR, exist_ok=True)

@app.on_event("startup")
def startup_event():
    load_files()

@app.post("/reload", response_model=DataResultModel)
def reload_files():
    load_files()
    return DataResultModel(source="reload", result="Reloaded all core files and index.")

@app.post("/get-data", response_model=DataResultModel)
async def get_data(payload: QueryRequest = Body(...)):
    query = payload.query.strip()
    page = max(payload.page or 1, 1)
    page_size = min(payload.page_size or MAX_RESULTS, ALLOWED_PAGE_SIZE)
    fields = [f.strip() for f in (payload.fields or "").split(",") if f.strip()] if payload.fields else None
    qlow = query.lower()
    if any(
        kw in qlow
        for kw in [
            "roll d10", "roll 1d10",
            "roll d6", "roll 1d6",
            "roll d100", "roll 1d100"
        ]
    ):
        if "d10" in qlow:
            return DataResultModel(source="dice", result={"roll": roll_d10()}, note="Rolled d10")
        if "d100" in qlow:
            return DataResultModel(source="dice", result={"roll": roll_d100()}, note="Rolled d100")
        if "d6" in qlow:
            return DataResultModel(source="dice", result={"roll": roll_d6()}, note="Rolled d6")
    if any(kw in qlow for kw in ["boot", "what is this", "intro"]):
        return DataResultModel(source="Bootloader.md", result=cache["bootloader"])
    index = cache.get("index")
    if index is not None and not index.empty:
        index_row = match_query(query, "LemDescription", index)
        core_file = index_row.get("Filename", "")
        if core_file:
            core_df = cache["cores"].get(core_file)
            if core_df is not None and "LemTrigger" in core_df.columns:
                core_row = match_query(query, "LemTrigger", core_df)
                sub_file = core_row.get("Filename", "")
                if sub_file:
                    sub_path = os.path.join(os.path.dirname(__file__), sub_file)
                    if os.path.exists(sub_path):
                        sub_df = pd.read_csv(sub_path, sep="\t")
                        match_row = match_query(query, "LemTrigger", sub_df) if "LemTrigger" in sub_df.columns else None
                        if match_row:
                            if fields:
                                match_row = {k: v for k, v in match_row.items() if k in fields}
                            return DataResultModel(source=sub_file, result=sanitize_for_json(match_row))
                        mask = sub_df.apply(lambda row: query.lower() in str(row).lower(), axis=1)
                        results = sub_df[mask] if not sub_df.empty else pd.DataFrame()
                        total = len(results)
                        if not results.empty:
                            start = (page - 1) * page_size
                            end = start + page_size
                            paged = results.iloc[start:end]
                            if fields:
                                paged = paged[fields]
                            out = paged.to_dict(orient="records")
                            note = f"{total} results; page {page}, page size {page_size}" + ("; truncated" if total > page_size else "")
                            return DataResultModel(source=sub_file, result=sanitize_for_json(out), note=note)
                        return DataResultModel(source=sub_file, result=[], note="No results found in sub-table.")
                    else:
                        return DataResultModel(source=core_file, result=core_row, note=f"Linked file {sub_file} not found.")
            if fields and isinstance(core_row, dict):
                core_row = {k: v for k, v in core_row.items() if k in fields}
            return DataResultModel(source=core_file, result=sanitize_for_json(core_row))
    # --- Hybrid Fallback: Directory-wide Scan ---
    data_folder = os.path.dirname(__file__)
    file, df = fallback_file_lookup(query, data_folder)
    if file and df is not None:
        # Return top N rows or best fuzzy matches
        mask = df.apply(lambda row: query.lower() in str(row).lower(), axis=1)
        results = df[mask] if not df.empty else pd.DataFrame()
        if not results.empty:
            return DataResultModel(source=file, result=sanitize_for_json(results.head(page_size).to_dict(orient="records")),
                                   note="Result from direct file scan (hybrid fallback).")
        # Fallback: Just return top N rows
        return DataResultModel(source=file, result=sanitize_for_json(df.head(page_size).to_dict(orient="records")),
                               note="Result from direct file scan, no strong row match.")
    return DataResultModel(source="none", result={}, note="No matching file or data found.")

# --- Remaining endpoints (unchanged) ---
@app.get("/roll", response_model=DiceResultModel)
def roll_dice(type: str = Query(..., description="Type of die: d6, d10, or d100")):
    t = type.lower()
    if t == "d6":
        return DiceResultModel(result=roll_d6())
    elif t == "d10":
        return DiceResultModel(result=roll_d10())
    elif t == "d100":
        return DiceResultModel(result=roll_d100())
    else:
        return DiceResultModel(error="Unsupported die type. Use d6, d10, or d100.")

@app.post("/combat/start", response_model=CombatResultModel)
def start_combat_endpoint(payload: CombatantsModel):
    try:
        result = start_combat(payload.combatants)
        return CombatResultModel(result=result)
    except Exception as e:
        return CombatResultModel(error=str(e))

@app.post("/combat/next", response_model=CombatResultModel)
def next_turn_endpoint():
    try:
        result = next_turn()
        return CombatResultModel(result=result)
    except Exception as e:
        return CombatResultModel(error=str(e))

@app.get("/combat/current", response_model=CombatResultModel)
def current_turn_endpoint():
    try:
        result = get_current_turn()
        return CombatResultModel(result=result)
    except Exception as e:
        return CombatResultModel(error=str(e))

@app.post("/combat/resolve", response_model=CombatResultModel)
def resolve_attack_endpoint(payload: CombatResolveModel):
    try:
        result = resolve_attack(payload.attacker, payload.target, payload.weapon, payload.armor)
        return CombatResultModel(result=result)
    except Exception as e:
        return CombatResultModel(error=str(e))

@app.post("/save-game")
def save_game(data: SaveGameRequest):
    path = os.path.join(SAVE_DIR, f"{data.game_id}.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data.state, f)
        return {"status": "saved", "game_id": data.game_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load-game")
def load_game(data: LoadGameRequest):
    path = os.path.join(SAVE_DIR, f"{data.game_id}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Game not found.")
    try:
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        return {"status": "loaded", "game_id": data.game_id, "state": state}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- System Health and Index Routes ---

@app.api_route("/", methods=["GET", "HEAD"], response_model=DataResultModel)
def root():
    logger.info("Root route accessed")
    return DataResultModel(
        source="root",
        result="Cyberpunk 2020 API is live",
        note="Try /get-data, /combat/start, or /roll?type=d10"
    )

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/list-files", response_model=DataResultModel)
def list_files():
    data_folder = os.path.dirname(__file__)
    files = sorted([f for f in os.listdir(data_folder) if f.endswith(".tsv") or f.endswith(".md")])
    return DataResultModel(source="list-files", result=files)


# --- CLI: TSV Validation Tool ---

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate all .tsv files in the directory")
    parser.add_argument("--folder", type=str, default=os.path.dirname(__file__), help="Folder to scan for .tsv files")
    args = parser.parse_args()

    tsv_files = glob.glob(os.path.join(args.folder, "*.tsv"))
    print(f"[INFO] Found {len(tsv_files)} TSV files in '{args.folder}'\n")

    for tsv in tsv_files:
        try:
            df = pd.read_csv(tsv, sep="\t")
            print(f"[OK] {os.path.basename(tsv)} — {df.shape[0]} rows, {df.shape[1]} columns")
        except Exception as e:
            print(f"[ERROR] {os.path.basename(tsv)} — {e}")


@app.get("/canon-map-keys")
def get_canon_map_keys():
    return {"roles": list(cache.get("canon_map", {}).keys())}