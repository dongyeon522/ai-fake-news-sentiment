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
import numpy as np
import re

from src.collector import NewsCollector
from src.preprocess import load_raw_articles, build_corpus
from src.sentiment import annotate_sentiment, annotate_sentiment_roberta
from src.evaluate import (
    evaluate_sentiment_distribution, 
    sentiment_to_label,
    precision_at_k,
    ndcg_at_k,
    mean_reciprocal_rank,
    evaluate_ranking
)
from src.ranking import TfidfRanker, BM25Ranker
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
    print("\n[1/6] Collecting news articles...")
    
    # MAX_PAGES setting (test mode: 3, production mode: 20 or 50)
    MAX_PAGES = 3  # Test mode (change to 20 or 50 for production)
    
    # Sentiment analysis method selection
    print("\nSelect sentiment analysis method:")
    print("  1. TextBlob (faster, rule-based)")
    print("  2. RoBERTa (slower, more accurate)")
    while True:
        choice = input("Enter choice (1 or 2, default: 1): ").strip()
        if choice == '' or choice == '1':
            SENTIMENT_METHOD = 'textblob'
            break
        elif choice == '2':
            SENTIMENT_METHOD = 'roberta'
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")
    
    collector = NewsCollector(api_key=api_key, max_pages=MAX_PAGES)
    print(f"Page limit: Max {MAX_PAGES} pages (max {MAX_PAGES * 10} articles per query)")
    
    # Check remaining requests
    remaining = collector.get_remaining_requests()
    if remaining == 0:
        print("Error: Daily request limit (5000) reached.")
        print("  Please try again tomorrow or use a paid plan.")
        sys.exit(1)
    
    # Date range: Fixed to 2024
    from_date = "2024-01-01"
    to_date = "2024-12-31"
    
    # Search queries
    queries = ["artificial intelligence", "deepfake", "fake news", "misinformation"]
    
    # Test mode: Use fewer months for faster testing
    # Set TEST_MODE=False for full year search
    TEST_MODE = False  # Full year search enabled
    if TEST_MODE:
        # Test mode: Only search last 3 months for faster testing
        from_date = "2024-10-01"
        print("  WARNING: TEST MODE - Searching only last 3 months (Oct-Dec 2024)")
    
    print(f"\nSearch period: {from_date} ~ {to_date}")
    print(f"Search queries: {', '.join(queries)}")
    
    # Adjust queries based on request limit
    if len(queries) > remaining:
        print(f"WARNING: Number of queries ({len(queries)}) exceeds remaining requests ({remaining}).")
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
            monthly=True  # Enabled for full year coverage (optimized with batch I/O)
        )
    except Exception as e:
        print(f"Error occurred during news collection: {e}")
        sys.exit(1)
    
    # 2. Data preprocessing
    print("\n[2/6] Preprocessing data...")
    try:
        df_processed = build_corpus(df_raw)
        print(f"{len(df_processed)} articles preprocessed.")
    except Exception as e:
        print(f"Error occurred during preprocessing: {e}")
        sys.exit(1)
    
    # 3. Sentiment analysis
    print("\n[3/6] Performing sentiment analysis...")
    print(f"  Using method: {SENTIMENT_METHOD.upper()}")
    try:
        if SENTIMENT_METHOD.lower() == 'roberta':
            df_sentiment = annotate_sentiment_roberta(df_processed)
        else:  # default to textblob
            df_sentiment = annotate_sentiment(df_processed)
        processed_file_path = get_filename_with_timestamp("data/processed/corpus_with_sentiment.csv")
        df_sentiment.to_csv(processed_file_path, index=False)
        print("Sentiment analysis completed.")
    except Exception as e:
        print(f"Error occurred during sentiment analysis: {e}")
        sys.exit(1)
    
    # 4. Sentiment evaluation
    print("\n[4/6] Evaluating sentiment results...")
    try:
        stats = evaluate_sentiment_distribution(df_sentiment)
        print("\nSentiment distribution statistics:")
        print(f"  Mean sentiment: {stats['mean_sentiment']:.3f}")
        print(f"  Standard deviation: {stats['std_sentiment']:.3f}")
        print(f"  Positive ratio: {stats['positive_ratio']:.2%}")
        print(f"  Negative ratio: {stats['negative_ratio']:.2%}")
        print(f"  Neutral ratio: {stats['neutral_ratio']:.2%}")
    except Exception as e:
        print(f"Error occurred during evaluation: {e}")
    
    # 5. Ranking experiment
    print("\n[5/6] Running ranking experiments...")
    try:
        # Prepare documents for ranking
        documents = df_sentiment["text"].tolist()
        queries_for_ranking = queries  # Use same queries as collection
        
        print(f"  Training rankers on {len(documents)} documents...")
        
        # Train TF-IDF ranker
        tfidf_ranker = TfidfRanker()
        tfidf_ranker.fit(documents)
        print("  TF-IDF ranker trained")
        
        # Train BM25 ranker (requires tokenized documents)
        tokenized_docs = [re.findall(r'\b\w+\b', doc.lower()) for doc in documents]
        bm25_ranker = BM25Ranker(tokenized_docs)
        print("  BM25 ranker trained")
        
        # Evaluate ranking for each query
        top_k = 10
        tfidf_results = []
        bm25_results = []
        
        print(f"\n  Evaluating ranking performance (top-{top_k}):")
        print("  " + "-" * 56)
        
        for query in queries_for_ranking:
            # TF-IDF ranking
            tfidf_ranked = tfidf_ranker.search(query, top_k=top_k)
            # Calculate relevance: 1 if query matches article's original query, 0 otherwise
            tfidf_relevance = []
            for idx, score in tfidf_ranked:
                article_query = df_sentiment.iloc[idx]["query"]
                # Check if article was collected with this query
                relevance = 1.0 if query.lower() in article_query.lower() or article_query.lower() in query.lower() else 0.0
                tfidf_relevance.append(relevance)
            
            # BM25 ranking
            query_tokens = re.findall(r'\b\w+\b', query.lower())
            bm25_ranked = bm25_ranker.search(query_tokens, top_k=top_k)
            bm25_relevance = []
            for idx, score in bm25_ranked:
                article_query = df_sentiment.iloc[idx]["query"]
                relevance = 1.0 if query.lower() in article_query.lower() or article_query.lower() in query.lower() else 0.0
                bm25_relevance.append(relevance)
            
            # Calculate metrics
            tfidf_metrics = evaluate_ranking(tfidf_relevance, top_k=top_k)
            bm25_metrics = evaluate_ranking(bm25_relevance, top_k=top_k)
            
            tfidf_results.append(tfidf_metrics)
            bm25_results.append(bm25_metrics)
            
            print(f"  Query: '{query}'")
            print(f"    TF-IDF - Precision@{top_k}: {tfidf_metrics['precision_at_k']:.3f}, "
                  f"NDCG@{top_k}: {tfidf_metrics['ndcg']:.3f}, MRR: {tfidf_metrics['mrr']:.3f}")
            print(f"    BM25  - Precision@{top_k}: {bm25_metrics['precision_at_k']:.3f}, "
                  f"NDCG@{top_k}: {bm25_metrics['ndcg']:.3f}, MRR: {bm25_metrics['mrr']:.3f}")
        
        # Calculate average metrics
        avg_tfidf = {
            'precision_at_k': np.mean([r['precision_at_k'] for r in tfidf_results]),
            'ndcg': np.mean([r['ndcg'] for r in tfidf_results]),
            'mrr': np.mean([r['mrr'] for r in tfidf_results])
        }
        avg_bm25 = {
            'precision_at_k': np.mean([r['precision_at_k'] for r in bm25_results]),
            'ndcg': np.mean([r['ndcg'] for r in bm25_results]),
            'mrr': np.mean([r['mrr'] for r in bm25_results])
        }
        
        print("\n  Average Ranking Performance:")
        print("  " + "-" * 56)
        print(f"  TF-IDF - Precision@{top_k}: {avg_tfidf['precision_at_k']:.3f}, "
              f"NDCG@{top_k}: {avg_tfidf['ndcg']:.3f}, MRR: {avg_tfidf['mrr']:.3f}")
        print(f"  BM25   - Precision@{top_k}: {avg_bm25['precision_at_k']:.3f}, "
              f"NDCG@{top_k}: {avg_bm25['ndcg']:.3f}, MRR: {avg_bm25['mrr']:.3f}")
        
        print("  Ranking experiments completed.")
    except Exception as e:
        print(f"Error occurred during ranking experiments: {e}")
        import traceback
        traceback.print_exc()
    
    # 6. Visualization
    print("\n[6/6] Generating visualization...")
    try:
        figure_file_path = get_filename_with_timestamp("data/figures/sentiment_over_time.png")
        plot_sentiment_over_time(df_sentiment, save_path=figure_file_path)
        print("Visualization completed.")
    except Exception as e:
        print(f"Error occurred during visualization: {e}")
    
    print("\n" + "=" * 60)
    print("All tasks completed!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - Raw data: {raw_file_path}")
    print(f"  - Processed data: {processed_file_path}")
    print(f"  - Visualization: {figure_file_path}")

if __name__ == "__main__":
    main()
