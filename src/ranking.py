# src/ranking.py
from typing import List, Tuple
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class TfidfRanker:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english", max_df=0.8, min_df=2)
        self.doc_vectors = None
        self.docs = None

    def fit(self, docs: List[str]):
        self.docs = docs
        self.doc_vectors = self.vectorizer.fit_transform(docs)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        q_vec = self.vectorizer.transform([query])
        scores = linear_kernel(q_vec, self.doc_vectors).flatten()
        ranked_indices = scores.argsort()[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in ranked_indices]

from rank_bm25 import BM25Okapi

class BM25Ranker:
    def __init__(self, tokenized_docs: List[List[str]]):
        self.bm25 = BM25Okapi(tokenized_docs)
        self.docs = tokenized_docs

    def search(self, query_tokens: List[str], top_k: int = 10):
        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = scores.argsort()[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in ranked_indices]
