from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
import spacy
from rapidfuzz import fuzz
from sklearn.metrics.pairwise import cosine_similarity

# Setup
app = FastAPI()
nlp = spacy.load("en_core_web_md")

# CORS for GPT/web calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory cache
cache = {
    "bootloader": "",
    "index": None,
    "cores": {}
}

# Synonyms (expand this as you go)
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
}

# Lemmatizer
def lemmatize(text):
    return " ".join(token.lemma_ for token in nlp(str(text)))

# Load all data on startup
@app.on_event("startup")
def load_files():
    data_folder = os.path.dirname(__file__)
    
    # Bootloader.md
    with open(os.path.join(data_folder, "Bootloader.md"), "r", encoding="utf-8") as f:
        cache["bootloader"] = f.read()

    # index.tsv
    index = pd.read_csv(os.path.join(data_folder, "index.tsv"), sep="\t")
    index["LemDescription"] = index["Description"].apply(lemmatize)
    cache["index"] = index

    # Load all *_core.tsv
    for file in os.listdir(data_folder):
        if file.endswith("_core.tsv"):
            df = pd.read_csv(os.path.join(data_folder, file), sep="\t")
            if "Trigger" in df.columns:
                df["LemTrigger"] = df["Trigger"].fillna("").apply(lemmatize)
            cache["cores"][file] = df

# Matching logic
def match_query(query, col, df):
    query_lem = lemmatize(query)

    for base, syns in synonyms.items():
        if any(term in query_lem for term in [base] + syns):
            match = df[df[col].str.contains(base)]
            if not match.empty:
                return match.iloc[0]

    fuzz_scores = df[col].apply(lambda x: fuzz.partial_ratio(query_lem, x))
    if fuzz_scores.max() > 85:
        return df.iloc[fuzz_scores.idxmax()]

    q_vec = nlp(query_lem).vector.reshape(1, -1)
    best = max(df.iterrows(), key=lambda row: cosine_similarity(q_vec, nlp(row[1][col]).vector.reshape(1, -1))[0][0])
    return best[1]

# API endpoint
@app.post("/get-data")
async def get_data(req: Request):
    body = await req.json()
    query = body.get("query", "")

    if "boot" in query or "what is this" in query or "intro" in query:
        return {"source": "Bootloader.md", "result": cache["bootloader"]}

    index_row = match_query(query, "LemDescription", cache["index"])
    core_file = index_row["Filename"]
    core_df = cache["cores"].get(core_file)

    if core_df is None or "LemTrigger" not in core_df.columns:
        return {"source": "index.tsv", "result": index_row.to_dict(), "note": "No valid core trigger."}

    core_row = match_query(query, "LemTrigger", core_df)

    # Optional sub-file
    if "Filename" in core_row and pd.notna(core_row["Filename"]):
        sub_file = core_row["Filename"]
        sub_path = os.path.join(os.path.dirname(__file__), sub_file)
        if os.path.exists(sub_path):
            sub_df = pd.read_csv(sub_path, sep="\t")
            return {"source": sub_file, "result": sub_df.to_dict(orient="records")}
        else:
            return {"source": core_file, "result": core_row.to_dict(), "note": f"Linked file {sub_file} not found."}

    return {"source": core_file, "result": core_row.to_dict()}
