# src/preprocess.py
import pandas as pd
import re

def load_raw_articles(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def build_corpus(df: pd.DataFrame) -> pd.DataFrame:
    df["title"] = df["title"].fillna("")
    df["description"] = df["description"].fillna("")
    df["content"] = df["content"].fillna("")
    df["text"] = (df["title"] + ". " + df["description"] + ". " + df["content"]).apply(clean_text)
    df = df[df["text"].str.len() > 0]
    return df
