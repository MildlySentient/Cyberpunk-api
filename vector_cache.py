# vector_cache.py
import numpy as np
import spacy

class VectorCache:
    def __init__(self):
        self.vectors = None
        self.texts = []
        self.df = None
        self.nlp = spacy.load("en_core_web_md")

    def preload_vectors(self, df):
        self.texts = [" ".join(str(x) for x in row) for _, row in df.iterrows()]
        self.vectors = np.array([self.nlp(text).vector for text in self.texts])
        # Normalize each vector to unit length
        if len(self.vectors) > 0:
            self.vectors /= np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-8
        self.df = df

    def find_best_match(self, query, top_k=1):
        if self.vectors is None or len(self.vectors) == 0:
            return []
        q_vec = self.nlp(query).vector
        norm = np.linalg.norm(q_vec) + 1e-8
        if norm == 0:
            return []
        q_vec /= norm
        sims = self.vectors @ q_vec
        top_idx = np.argsort(-sims)[:top_k]
        return [(i, float(sims[i])) for i in top_idx]