# src/sentiment.py
from textblob import TextBlob
import pandas as pd
from typing import Optional

# TextBlob version
def get_polarity(text: str) -> float:
    if not isinstance(text, str) or not text.strip():
        return 0.0
    blob = TextBlob(text)
    return blob.sentiment.polarity  # Range: -1 to 1

def annotate_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    df["sentiment"] = df["text"].apply(get_polarity)
    return df

# RoBERTa version
def get_polarity_roberta(text: str, model=None, tokenizer=None) -> float:
    """
    Get sentiment polarity using RoBERTa model.
    
    Args:
        text: Input text
        model: Pre-loaded RoBERTa model (optional, will load if None)
        tokenizer: Pre-loaded tokenizer (optional, will load if None)
    
    Returns:
        Sentiment polarity score in range [-1, 1]
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0
    
    try:
        from transformers import pipeline
        
        # Use sentiment analysis pipeline
        # This will automatically load the model on first call
        if not hasattr(get_polarity_roberta, '_pipeline'):
            get_polarity_roberta._pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=-1  # Use CPU (-1) or GPU (0, 1, ...)
            )
        
        # Truncate text if too long (RoBERTa has max length of 512 tokens)
        max_length = 512
        if len(text) > max_length * 4:  # Rough estimate: 4 chars per token
            text = text[:max_length * 4]
        
        result = get_polarity_roberta._pipeline(text)[0]
        
        # Convert label to polarity score
        # RoBERTa returns: {'label': 'POSITIVE'/'NEGATIVE'/'NEUTRAL' or 'positive'/'negative'/'neutral', 'score': 0.0-1.0}
        label = result['label'].upper()  # Normalize to uppercase
        score = result['score']
        
        if label == 'POSITIVE' or label == 'LABEL_2':  # Some models use LABEL_2 for positive
            return score  # 0.0 to 1.0
        elif label == 'NEGATIVE' or label == 'LABEL_0':  # Some models use LABEL_0 for negative
            return -score  # -1.0 to 0.0
        else:  # NEUTRAL or LABEL_1
            # For neutral, return a small value based on score (closer to 0)
            # This allows some differentiation between neutral texts
            return 0.0
            
    except ImportError:
        raise ImportError(
            "transformers library is required for RoBERTa sentiment analysis. "
            "Install it with: pip install transformers torch"
        )
    except Exception as e:
        print(f"Warning: RoBERTa sentiment analysis failed: {e}")
        return 0.0

def annotate_sentiment_roberta(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Annotate sentiment using RoBERTa model.
    
    Args:
        df: DataFrame with text column
        text_col: Name of the text column (default: "text")
    
    Returns:
        DataFrame with added "sentiment" column
    """
    print("Loading RoBERTa model for sentiment analysis...")
    print("(This may take a moment on first run)")
    
    # Initialize pipeline by calling get_polarity_roberta once to trigger model loading
    # This ensures the pipeline is loaded before applying to all rows
    if not hasattr(get_polarity_roberta, '_pipeline') or get_polarity_roberta._pipeline is None:
        # Trigger pipeline initialization with a dummy call
        sample_text = df[text_col].iloc[0] if len(df) > 0 else "test"
        _ = get_polarity_roberta(sample_text)
    
    df = df.copy()
    df["sentiment"] = df[text_col].apply(get_polarity_roberta)
    return df
