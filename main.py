#!/usr/bin/env python3
"""
AI Fake News Sentiment Analysis - Main execution script
"""
import os
import sys
from datetime import datetime, timedelta
from dotenv import load_dotenv
import glob
import pandas as pd

from src.collector import NewsCollector
from src.preprocess import load_raw_articles, build_corpus
from src.sentiment import annotate_sentiment
from src.evaluate import evaluate_sentiment_distribution, sentiment_to_label
from src.visualize import plot_sentiment_over_time

def main():
    """Main execution function"""
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("NYT_API_KEY")
    
    if not api_key:
        print("Error: NYT_API_KEY is not set.")
        print("Please set NYT_API_KEY in .env file.")
        sys.exit(1)
    
    # Create directories
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/figures", exist_ok=True)
    
    # Function to add timestamp to filename
    def get_filename_with_timestamp(base_path: str, prefix: str = "") -> str:
        """
        Add date and number to filename.
        Example: articles_20241123_001.csv
        """
        directory = os.path.dirname(base_path)
        base_name = os.path.basename(base_path)
        name, ext = os.path.splitext(base_name)
        
        # Date format: YYYYMMDD
        date_str = datetime.now().strftime("%Y%m%d")
        
        # Count existing files with same date to assign number
        pattern = os.path.join(directory, f"{name}_{date_str}_*{ext}")
        existing_files = glob.glob(pattern)
        file_num = len(existing_files) + 1
        
        # Generate filename: name_YYYYMMDD_NNN.ext
        new_filename = f"{name}_{date_str}_{file_num:03d}{ext}"
        return os.path.join(directory, new_filename)
    
    print("=" * 60)
    print("AI Fake News Sentiment Analysis")
    print("=" * 60)
    
    # 1. News collection
    print("\n[1/5] Collecting news articles...")
    
    # MAX_PAGES setting (test mode: 3, production mode: 20 or 50)
    MAX_PAGES = 3  # Test mode (change to 20 or 50 for production)
    
    collector = NewsCollector(api_key=api_key, max_pages=MAX_PAGES)
    print(f"📄 Page limit: Max {MAX_PAGES} pages (max {MAX_PAGES * 10} articles per query)")
    
    # Check remaining requests
    remaining = collector.get_remaining_requests()
    if remaining == 0:
        print("✗ Daily request limit (5000) reached.")
        print("  Please try again tomorrow or use a paid plan.")
        sys.exit(1)
    
    # Date range: Fixed to 2024
    from_date = "2024-01-01"
    to_date = "2024-12-31"
    
    # Search queries
    queries = ["artificial intelligence", "deepfake", "fake news", "misinformation"]
    
    print(f"\n📅 Search period: {from_date} ~ {to_date}")
    print(f"🔍 Search queries: {', '.join(queries)}")
    
    # Adjust queries based on request limit
    if len(queries) > remaining:
        print(f"⚠ Warning: Number of queries ({len(queries)}) exceeds remaining requests ({remaining}).")
        queries = queries[:remaining]
        print(f"  Processing {len(queries)} queries only.")
    
    try:
        raw_file_path = get_filename_with_timestamp("data/raw/articles.csv")
        # Test mode: 10 articles per search
        page_size_per_query = 10
        df_raw = collector.collect_and_save(
            queries=queries,
            from_date=from_date,
            to_date=to_date,
            out_path=raw_file_path,
            page_size=page_size_per_query,
            monthly=True  # Execute queries monthly
        )
    except Exception as e:
        print(f"✗ Error occurred during news collection: {e}")
        sys.exit(1)
    
    # 2. Data preprocessing
    print("\n[2/5] Preprocessing data...")
    try:
        df_processed = build_corpus(df_raw)
        print(f"✓ {len(df_processed)} articles preprocessed.")
    except Exception as e:
        print(f"✗ Error occurred during preprocessing: {e}")
        sys.exit(1)
    
    # 3. Sentiment analysis
    print("\n[3/5] Performing sentiment analysis...")
    try:
        df_sentiment = annotate_sentiment(df_processed)
        processed_file_path = get_filename_with_timestamp("data/processed/corpus_with_sentiment.csv")
        df_sentiment.to_csv(processed_file_path, index=False)
        print("✓ Sentiment analysis completed.")
    except Exception as e:
        print(f"✗ Error occurred during sentiment analysis: {e}")
        sys.exit(1)
    
    # 4. Evaluation
    print("\n[4/5] Evaluating results...")
    try:
        stats = evaluate_sentiment_distribution(df_sentiment)
        print("\nSentiment distribution statistics:")
        print(f"  Mean sentiment: {stats['mean_sentiment']:.3f}")
        print(f"  Standard deviation: {stats['std_sentiment']:.3f}")
        print(f"  Positive ratio: {stats['positive_ratio']:.2%}")
        print(f"  Negative ratio: {stats['negative_ratio']:.2%}")
        print(f"  Neutral ratio: {stats['neutral_ratio']:.2%}")
    except Exception as e:
        print(f"✗ Error occurred during evaluation: {e}")
    
    # 5. Visualization
    print("\n[5/5] Generating visualization...")
    try:
        figure_file_path = get_filename_with_timestamp("data/figures/sentiment_over_time.png")
        plot_sentiment_over_time(df_sentiment, save_path=figure_file_path)
        print("✓ Visualization completed.")
    except Exception as e:
        print(f"✗ Error occurred during visualization: {e}")
    
    print("\n" + "=" * 60)
    print("All tasks completed!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - Raw data: {raw_file_path}")
    print(f"  - Processed data: {processed_file_path}")
    print(f"  - Visualization: {figure_file_path}")

if __name__ == "__main__":
    main()
