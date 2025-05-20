import logging
import os
import json
from typing import List, Union, Optional, Dict, Any

import pandas as pd
import spacy
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from rapidfuzz import fuzz
from sklearn.metrics.pairwise import cosine_similarity

# --- Combat modules (must exist in your repo) ---
from cyberpunk2020_engine import roll_d6, roll_d10, roll_d100
from combat_tracker import start_combat, next_turn, get_current_turn
from combat_resolution import resolve_attack

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cyberpunk_api")

# --- FastAPI App ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourfrontend.com"],  # TODO: Set your real frontend here before launch!
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Cache ---
cache = {
    "bootloader": "",
    "index": None,
    "cores": {}
}

# --- Synonym dictionary ---
synonyms = {
    "steal": ["rob", "robbery", "theft"],
    "fight": ["attack", "punch", "strike"],
    "alarm": ["alert", "siren"],
    "campaign": ["operation", "mission", "initiative", "drive", "crusade", "push"],
    "handout": ["flyer", "pamphlet", "brochure", "leaflet", "giveaway", "document"],
    "asset": ["resource", "property", "capital", "benefit", "advantage", "holding"],
    "digital": ["electronic", "virtual", "binary", "online", "computerized", "data-driven"],
    "fail": ["crash", "break", "malfunction", "collapse", "falter", "misfire"]
}

# --- spaCy model with fallback ---
try:
    logger.info("Loading spaCy model: en_core_web_md")
    nlp = spacy.load("en_core_web_md")
except Exception as e:
    logger.error("spaCy model load error: %s", e)
    nlp = None

def lemmatize(text):
    if not nlp or pd.isna(text):
        return str(text) if text is not None else ""
    return " ".join(token.lemma_ for token in nlp(str(text)))

# --- File Loading Function ---
def load_files():
    logger.info("Loading Bootloader.md, index.tsv, and all _core.tsv files...")
    data_folder = os.path.dirname(__file__)
    # Bootloader
    try:
        with open(os.path.join(data_folder, "Bootloader.md"), "r", encoding="utf-8") as f:
            cache["bootloader"] = f.read()
        logger.info("Loaded Bootloader.md")
    except Exception as e:
        logger.error("Failed to load Bootloader.md: %s", e)
        cache["bootloader"] = "[ERROR] Bootloader not found."
    # Index
    try:
        index_path = os.path.join(data_folder, "index.tsv")
        index = pd.read_csv(index_path, sep="\t")
        index["Description"] = index["Description"].fillna("").astype(str)
        index["LemDescription"] = index["Description"].apply(lemmatize)
        cache["index"] = index
        logger.info("Loaded index.tsv")
    except Exception as e:
        logger.error("Failed to load index.tsv: %s", e)
        cache["index"] = pd.DataFrame([{"Filename":"", "Description":"", "LemDescription":""}])
    # Cores
    cache["cores"] = {}
    for file in os.listdir(data_folder):
        if file.endswith("_core.tsv"):
            try:
                df = pd.read_csv(os.path.join(data_folder, file), sep="\t")
                if "Trigger" in df.columns:
                    df["Trigger"] = df["Trigger"].fillna("").astype(str)
                    df["LemTrigger"] = df["Trigger"].apply(lemmatize)
                else:
                    df["LemTrigger"] = ""
                cache["cores"][file] = df
                logger.info("Loaded %s", file)
            except Exception as e:
                logger.error("Failed to load %s: %s", file, e)
    logger.info("Cores loaded: %s", list(cache["cores"].keys()))

# --- Query Matching ---
def match_query(query, col, df):
    if df is None or df.empty or col not in df.columns:
        logger.warning("match_query: DataFrame missing/empty or column '%s' not found", col)
        return {}
    query_lem = lemmatize(query)
    # Synonym match
    for base, syns in synonyms.items():
        if any(term in query_lem for term in [base] + syns):
            mask = df[col].str.contains(base, na=False)
            match = df[mask]
            if not match.empty:
                logger.info("match_query: Synonym match for '%s'", base)
                return match.iloc[0].to_dict()
    # Fuzzy match
    fuzz_scores = df[col].apply(lambda x: fuzz.partial_ratio(query_lem, str(x)))
    if fuzz_scores.max() > 85:
        logger.info("match_query: Fuzzy match found")
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
            logger.info("match_query: Cosine similarity match")
            return best
    logger.warning("match_query: No match found for query '%s'", query)
    return {}

# --- Pydantic Models: Input ---
class QueryRequest(BaseModel):
    query: str = Field(..., description="Text query to search for relevant game data")

class CombatantsModel(BaseModel):
    combatants: List[str] = Field(..., description="List of combatant names")

class CombatResolveModel(BaseModel):
    attacker: str
    target: str
    weapon: str
    armor: str

# --- Game Save/Load Models ---
class SaveGameRequest(BaseModel):
    game_id: str
    state: dict

class LoadGameRequest(BaseModel):
    game_id: str

# --- Pydantic Models: Output ---
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

# --- Startup: load everything on boot ---
@app.on_event("startup")
def startup_event():
    load_files()

# --- Hot reload endpoint ---
@app.post("/reload", response_model=DataResultModel)
def reload_files():
    load_files()
    logger.info("Hot reload complete.")
    return DataResultModel(source="reload", result="Reloaded all core files and index.")

# --- API Endpoints ---

@app.post("/get-data", response_model=DataResultModel)
async def get_data(payload: QueryRequest):
    query = payload.query
    if any(kw in query.lower() for kw in ["boot", "what is this", "intro"]):
        return DataResultModel(source="Bootloader.md", result=cache["bootloader"])
    index = cache.get("index")
    index_row = match_query(query, "LemDescription", index)
    core_file = index_row.get("Filename", "")
    if not core_file:
        return DataResultModel(source="index.tsv", result=index_row, note="No matching index.")
    core_df = cache["cores"].get(core_file)
    if core_df is None or "LemTrigger" not in core_df.columns:
        return DataResultModel(source="index.tsv", result=index_row, note="No valid core trigger.")
    core_row = match_query(query, "LemTrigger", core_df)
    sub_file = core_row.get("Filename", "")
    if sub_file:
        sub_path = os.path.join(os.path.dirname(__file__), sub_file)
        if os.path.exists(sub_path):
            sub_df = pd.read_csv(sub_path, sep="\t")
            return DataResultModel(source=sub_file, result=sub_df.to_dict(orient="records"))
        else:
            return DataResultModel(source=core_file, result=core_row, note=f"Linked file {sub_file} not found.")
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
        logger.error("Combat start error: %s", e)
        return CombatResultModel(error=str(e))

@app.post("/combat/next", response_model=CombatResultModel)
def next_turn_endpoint():
    try:
        result = next_turn()
        return CombatResultModel(result=result)
    except Exception as e:
        logger.error("Combat next error: %s", e)
        return CombatResultModel(error=str(e))

@app.get("/combat/current", response_model=CombatResultModel)
def current_turn_endpoint():
    try:
        result = get_current_turn()
        return CombatResultModel(result=result)
    except Exception as e:
        logger.error("Combat current error: %s", e)
        return CombatResultModel(error=str(e))

@app.post("/combat/resolve", response_model=CombatResultModel)
def resolve_attack_endpoint(payload: CombatResolveModel):
    try:
        result = resolve_attack(payload.attacker, payload.target, payload.weapon, payload.armor)
        return CombatResultModel(result=result)
    except Exception as e:
        logger.error("Combat resolve error: %s", e)
        return CombatResultModel(error=str(e))

# --- JSON Save/Load Endpoints ---

@app.post("/save-game")
def save_game(data: SaveGameRequest):
    path = os.path.join(SAVE_DIR, f"{data.game_id}.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data.state, f)
        return {"status": "saved", "game_id": data.game_id}
    except Exception as e:
        logger.error("Save game error: %s", e)
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
        logger.error("Load game error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
