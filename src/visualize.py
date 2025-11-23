# src/visualize.py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
from datetime import datetime

def plot_sentiment_over_time(df: pd.DataFrame, time_col: str = "published_at", freq: str = "ME", 
                             save_path: str = "data/figures/sentiment_over_time.png"):
    """
    Visualize sentiment changes over time
    
    Args:
        df: DataFrame with sentiment analysis results
        time_col: Time column name
        freq: Resampling frequency (ME=month end, D=day, W=week)
        save_path: Save path
    """
    # Convert date column
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    
    # Remove timezone (if UTC exists)
    if df[time_col].dt.tz is not None:
        df[time_col] = df[time_col].dt.tz_localize(None)
    
    # Fixed to full 2024 range
    date_start = pd.Timestamp('2024-01-01')
    date_end = pd.Timestamp('2024-12-31')
    
    # Filter data to 2024 range
    df = df[(df[time_col] >= date_start) & (df[time_col] <= date_end)]
    
    # Always plot monthly
    freq = "ME"  # Month End
    
    # Generate full month range for 2024
    date_range = pd.date_range(start=date_start, end=date_end, freq='MS')  # Start of each month
    date_range_end = pd.date_range(start=date_start, end=date_end, freq='ME')  # End of each month
    
    # Resample (include full 2024 month range)
    df_indexed = df.set_index(time_col)
    grouped = df_indexed.resample(freq)["sentiment"].mean().reset_index()
    
    # Reindex to include all 2024 months
    full_month_range = pd.date_range(start=date_start, end=date_end, freq='ME')
    grouped = grouped.set_index(time_col).reindex(full_month_range).reset_index()
    grouped.columns = [time_col, "sentiment"]
    
    # Visualization
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=grouped, x=time_col, y="sentiment", marker="o", linewidth=2, markersize=8)
    
    # Title and labels (reflect search conditions)
    plt.title("Average Sentiment about AI Misinformation Over Time\n"
              "(Search: artificial intelligence, deepfake, fake news, misinformation | 2024)", 
              fontsize=14, pad=20)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Average Sentiment Polarity", fontsize=12)
    
    # X-axis settings: display monthly from January to December 2024
    ax = plt.gca()
    
    # Set x-axis range to full 2024
    ax.set_xlim([date_start, date_end])
    
    # Set monthly ticks (middle of each month)
    month_ticks = pd.date_range(start=date_start, end=date_end, freq='MS') + pd.Timedelta(days=15)
    ax.set_xticks(month_ticks)
    
    # Monthly labels (YYYY-MM format)
    ax.set_xticklabels([d.strftime('%Y-%m') for d in month_ticks], rotation=45, ha='right')
    
    # Set date formatter
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # Add grid
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add neutral line
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Neutral')
    
    # Legend
    plt.legend()
    
    plt.tight_layout()
    
    # Create save directory
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save to file
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Visualization saved: {save_path}")
    
    # Also display on screen
    plt.show()
