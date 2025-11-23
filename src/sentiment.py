# src/sentiment.py
from textblob import TextBlob
import pandas as pd

def get_polarity(text: str) -> float:
    if not isinstance(text, str) or not text.strip():
        return 0.0
    blob = TextBlob(text)
    return blob.sentiment.polarity  # Range: -1 to 1

def annotate_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    df["sentiment"] = df["text"].apply(get_polarity)
    return df
