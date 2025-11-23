# src/evaluate.py
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import List, Dict
import numpy as np

def evaluate_sentiment_classification(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    """
    Evaluate sentiment classification performance
    
    Args:
        y_true: List of true labels (e.g., ['positive', 'negative', 'neutral'])
        y_pred: List of predicted labels
    
    Returns:
        Dictionary of evaluation metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def sentiment_to_label(polarity: float, threshold: float = 0.1) -> str:
    """
    Convert sentiment polarity to label
    
    Args:
        polarity: Sentiment polarity value (-1 to 1)
        threshold: Threshold for classifying as neutral
    
    Returns:
        'positive', 'negative', or 'neutral'
    """
    if polarity > threshold:
        return 'positive'
    elif polarity < -threshold:
        return 'negative'
    else:
        return 'neutral'

def evaluate_ranking(relevance_scores: List[float], top_k: int = 10) -> Dict[str, float]:
    """
    Evaluate ranking performance
    
    Args:
        relevance_scores: List of relevance scores
        top_k: Top k items to evaluate
    
    Returns:
        Dictionary of evaluation metrics (MRR, NDCG, etc.)
    """
    if len(relevance_scores) == 0:
        return {'mrr': 0.0, 'ndcg': 0.0, 'mean_score': 0.0}
    
    scores = np.array(relevance_scores[:top_k])
    
    # Mean Reciprocal Rank (MRR)
    # Reciprocal rank of first relevant document
    first_relevant_idx = np.where(scores > 0)[0]
    mrr = 1.0 / (first_relevant_idx[0] + 1) if len(first_relevant_idx) > 0 else 0.0
    
    # Normalized Discounted Cumulative Gain (NDCG)
    # Simplified version: normalize scores
    ideal_scores = np.sort(scores)[::-1]
    dcg = np.sum(scores / np.log2(np.arange(2, len(scores) + 2)))
    idcg = np.sum(ideal_scores / np.log2(np.arange(2, len(ideal_scores) + 2)))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    
    return {
        'mrr': mrr,
        'ndcg': ndcg,
        'mean_score': float(np.mean(scores))
    }

def evaluate_sentiment_distribution(df: pd.DataFrame) -> Dict[str, float]:
    """
    Evaluate sentiment distribution statistics
    
    Args:
        df: DataFrame with sentiment column
    
    Returns:
        Sentiment distribution statistics
    """
    if 'sentiment' not in df.columns:
        raise ValueError("DataFrame does not have 'sentiment' column.")
    
    sentiment_values = df['sentiment'].dropna()
    
    return {
        'mean_sentiment': float(sentiment_values.mean()),
        'std_sentiment': float(sentiment_values.std()),
        'positive_ratio': float((sentiment_values > 0.1).sum() / len(sentiment_values)),
        'negative_ratio': float((sentiment_values < -0.1).sum() / len(sentiment_values)),
        'neutral_ratio': float(((sentiment_values >= -0.1) & (sentiment_values <= 0.1)).sum() / len(sentiment_values))
    }
