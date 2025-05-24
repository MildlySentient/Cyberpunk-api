# vector_cache.py
import numpy as np

class VectorCache:
    def __init__(self):
        self.vectors = None
        self.texts = []
        self.df = None

    def preload_vectors(self, df):
        import spacy
        nlp = spacy.load("en_core_web_md")
        self.texts = [" ".join(str(x) for x in row) for _, row in df.iterrows()]
        self.vectors = np.array([nlp(text).vector for text in self.texts])
        self.df = df

    def find_best_match(self, query, top_k=1):
        import spacy
        nlp = spacy.load("en_core_web_md")
        q_vec = nlp(query).vector
        if self.vectors is None or len(self.vectors) == 0:
            return []
        sims = self.vectors @ q_vec / (np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(q_vec) + 1e-8)
        # Top-k indices and scores
        top_idx = np.argsort(-sims)[:top_k]
        return [(i, float(sims[i])) for i in top_idx]