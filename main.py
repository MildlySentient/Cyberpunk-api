import logging
import os
import json
from typing import List, Union, Optional, Dict, Any

import pandas as pd
import spacy
from fastapi import FastAPI, Query, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from rapidfuzz import fuzz
from sklearn.metrics.pairwise import cosine_similarity

# --- Combat modules ---
from cyberpunk2020_engine import roll_d6, roll_d10, roll_d100
from combat_tracker import start_combat, next_turn, get_current_turn
from combat_resolution import resolve_attack

# --- Config ---
MAX_RESULTS = 25  # Max results per response (override via page_size)
ALLOWED_PAGE_SIZE = 50  # Hard cap

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
    "steal": ["rob", "robbery", "theft"],
    "fight": ["attack", "punch", "strike"],
    "alarm": ["alert", "siren"],
    # ... add more as needed
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

# --- File Loader ---
def load_files():
    data_folder = os.path.dirname(__file__)
    try:
        with open(os.path.join(data_folder, "Bootloader.md"), "r", encoding="utf-8") as f:
            cache["bootloader"] = f.read()
    except Exception as e:
        cache["bootloader"] = "[ERROR] Bootloader.md not found."
    try:
        index_path = os.path.join(data_folder, "index.tsv")
        index = pd.read_csv(index_path, sep="\t")
        index["Description"] = index["Description"].fillna("").astype(str)
        index["LemDescription"] = index["Description"].apply(lemmatize)
        cache["index"] = index
    except Exception as e:
        cache["index"] = pd.DataFrame([{"Filename": "", "Description": "", "LemDescription": ""}])
    cache["cores"] = {}
    for file in os.listdir(data_folder):
        if file.endswith("_core.tsv"):
            try:
                df = pd.read_csv(os.path.join(data_folder, file), sep="\t")
                df["Trigger"] = df.get("Trigger", "").fillna("").astype(str)
                df["LemTrigger"] = df["Trigger"].apply(lemmatize)
                cache["cores"][file] = df
            except Exception:
                pass

# --- Query Matching ---
def match_query(query, col, df):
    if df is None or df.empty or col not in df.columns:
        return {}
    query_lem = lemmatize(query)
    # Synonym match
    for base, syns in synonyms.items():
        if any(term in query_lem for term in [base] + syns):
            mask = df[col].str.contains(base, na=False)
            match = df[mask]
            if not match.empty:
                return match.iloc[0].to_dict()
    # Fuzzy match
    fuzz_scores = df[col].apply(lambda x: fuzz.partial_ratio(query_lem, str(x)))
    if fuzz_scores.max() > 85:
        return df.iloc[fuzz_scores.idxmax()].to_dict()
    # Cosine similarity
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

# --- Models ---
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

# --- Save folder ---
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
    """
    Flexible search endpoint with pagination and field filtering.
    Dice intent (roll d10, roll d6, roll d100, etc) is handled directly.
    """
    query = payload.query.strip()
    page = max(payload.page or 1, 1)
    page_size = min(payload.page_size or MAX_RESULTS, ALLOWED_PAGE_SIZE)
    fields = [f.strip() for f in (payload.fields or "").split(",") if f.strip()] if payload.fields else None

    # ---- DICE SHORT-CIRCUIT BLOCK ----
    qlow = query.lower()
    # Supported patterns: "roll d10", "roll 1d6", "roll d100", etc.
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
    # ---- END DICE SHORT-CIRCUIT ----

    # Handle intro
    if any(kw in qlow for kw in ["boot", "what is this", "intro"]):
        return DataResultModel(source="Bootloader.md", result=cache["bootloader"])
    index = cache.get("index")
    if index is None or index.empty:
        return DataResultModel(source="index.tsv", result={}, note="Index is empty or not loaded.")

    # Find best match in index
    index_row = match_query(query, "LemDescription", index)
    core_file = index_row.get("Filename", "")
    if not core_file:
        return DataResultModel(source="index.tsv", result=index_row, note="No matching index.")

    core_df = cache["cores"].get(core_file)
    if core_df is None or "LemTrigger" not in core_df.columns:
        return DataResultModel(source="index.tsv", result=index_row, note="No valid core trigger.")

    # Find best match in core table
    core_row = match_query(query, "LemTrigger", core_df)
    sub_file = core_row.get("Filename", "")

    # If there’s a subfile, try to load it and search
    if sub_file:
        sub_path = os.path.join(os.path.dirname(__file__), sub_file)
        if os.path.exists(sub_path):
            sub_df = pd.read_csv(sub_path, sep="\t")
            # Try to match the query to a row in sub_df
            match_row = match_query(query, "LemTrigger", sub_df) if "LemTrigger" in sub_df.columns else None
            if match_row:
                if fields:
                    match_row = {k: v for k, v in match_row.items() if k in fields}
                return DataResultModel(source=sub_file, result=match_row)
            # Fallback: list mode
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
                return DataResultModel(source=sub_file, result=out, note=note)
            return DataResultModel(source=sub_file, result=[], note="No results found in sub-table.")
        else:
            return DataResultModel(source=core_file, result=core_row, note=f"Linked file {sub_file} not found.")

    # No subfile: just return matched row, possibly filtered
    if fields and isinstance(core_row, dict):
        core_row = {k: v for k, v in core_row.items() if k in fields}
    return DataResultModel(source=core_file, result=core_row)

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