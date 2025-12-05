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

def precision_at_k(relevance_scores: List[float], k: int = 10) -> float:
    """
    Calculate Precision@K
    
    Args:
        relevance_scores: List of relevance scores (binary: 1 for relevant, 0 for non-relevant)
        k: Top k items to evaluate
    
    Returns:
        Precision@K score
    """
    if len(relevance_scores) == 0 or k == 0:
        return 0.0
    
    top_k_scores = relevance_scores[:k]
    relevant_count = sum(1 for score in top_k_scores if score > 0)
    return relevant_count / min(k, len(relevance_scores))

def ndcg_at_k(relevance_scores: List[float], k: int = 10) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain@K
    
    Args:
        relevance_scores: List of relevance scores
        k: Top k items to evaluate
    
    Returns:
        NDCG@K score
    """
    if len(relevance_scores) == 0 or k == 0:
        return 0.0
    
    scores = np.array(relevance_scores[:k])
    
    # Calculate DCG
    dcg = np.sum(scores / np.log2(np.arange(2, len(scores) + 2)))
    
    # Calculate IDCG (ideal DCG)
    ideal_scores = np.sort(relevance_scores)[::-1][:k]
    idcg = np.sum(ideal_scores / np.log2(np.arange(2, len(ideal_scores) + 2)))
    
    return dcg / idcg if idcg > 0 else 0.0

def mean_reciprocal_rank(relevance_scores_list: List[List[float]]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR)
    
    Args:
        relevance_scores_list: List of relevance score lists (one per query)
    
    Returns:
        Mean Reciprocal Rank
    """
    if len(relevance_scores_list) == 0:
        return 0.0
    
    reciprocal_ranks = []
    for relevance_scores in relevance_scores_list:
        scores = np.array(relevance_scores)
        # Find first relevant item (score > 0)
        relevant_indices = np.where(scores > 0)[0]
        if len(relevant_indices) > 0:
            rank = relevant_indices[0] + 1  # 1-indexed
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
    
    return float(np.mean(reciprocal_ranks))

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
        return {'mrr': 0.0, 'ndcg': 0.0, 'precision_at_k': 0.0, 'mean_score': 0.0}
    
    scores = np.array(relevance_scores[:top_k])
    
    # Mean Reciprocal Rank (MRR)
    # Reciprocal rank of first relevant document
    first_relevant_idx = np.where(scores > 0)[0]
    mrr = 1.0 / (first_relevant_idx[0] + 1) if len(first_relevant_idx) > 0 else 0.0
    
    # Normalized Discounted Cumulative Gain (NDCG)
    ndcg = ndcg_at_k(relevance_scores, top_k)
    
    # Precision@K
    precision = precision_at_k(relevance_scores, top_k)
    
    return {
        'mrr': mrr,
        'ndcg': ndcg,
        'precision_at_k': precision,
        'mean_score': float(np.mean(scores))
    }

def evaluate_sentiment_distribution(df: pd.DataFrame, pos_th: float = 0.1, neg_th: float = -0.1) -> Dict[str, float]:
    """
    Evaluate sentiment distribution statistics
    
    Args:
        df: DataFrame with sentiment column
        pos_th: Positive threshold (default: 0.1)
        neg_th: Negative threshold (default: -0.1)
    
    Returns:
        Sentiment distribution statistics
    """
    if 'sentiment' not in df.columns:
        raise ValueError("DataFrame does not have 'sentiment' column.")
    
    sentiment_values = df['sentiment'].dropna()
    
    if len(sentiment_values) == 0:
        return {
            'mean_sentiment': 0.0,
            'std_sentiment': 0.0,
            'positive_ratio': 0.0,
            'negative_ratio': 0.0,
            'neutral_ratio': 0.0
        }
    
    return {
        'mean_sentiment': float(sentiment_values.mean()),
        'std_sentiment': float(sentiment_values.std()),
        'positive_ratio': float((sentiment_values > pos_th).sum() / len(sentiment_values)),
        'negative_ratio': float((sentiment_values < neg_th).sum() / len(sentiment_values)),
        'neutral_ratio': float(((sentiment_values >= neg_th) & (sentiment_values <= pos_th)).sum() / len(sentiment_values))
    }
