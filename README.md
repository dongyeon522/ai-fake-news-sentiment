# AI Fake News Sentiment Analysis

A sentiment analysis and ranking pipeline for AI-related fake news articles.

## Project Overview

This project collects AI-related news articles, performs sentiment analysis, and ranks them by relevance. It uses the New York Times Article Search API to gather articles and analyzes sentiment using TextBlob.

## Project Structure

```
ai-fake-news-sentiment/
├── data/
│   ├── raw/          # Raw news data
│   ├── processed/    # Processed data
│   └── figures/      # Generated visualizations
├── src/
│   ├── collector.py  # News collector
│   ├── preprocess.py # Text preprocessing
│   ├── sentiment.py  # Sentiment analysis
│   ├── ranking.py    # Ranking algorithms (TF-IDF, BM25)
│   ├── evaluate.py   # Evaluation metrics
│   └── visualize.py  # Visualization
├── main.py           # Main execution script
└── requirements.txt
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ai-fake-news-sentiment
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Set NYT_API_KEY in .env file
```

## Usage

### Run the complete pipeline:

```bash
python main.py
```

This will:
1. Collect news articles from NYT API (monthly search mode)
2. Preprocess the data
3. Perform sentiment analysis
4. Evaluate results
5. Generate visualizations

### Use individual modules:

#### 1. News Collection

```python
from src.collector import NewsCollector
from dotenv import load_dotenv
import os

load_dotenv()
collector = NewsCollector(api_key=os.getenv("NYT_API_KEY"))

queries = ["artificial intelligence", "deepfake", "fake news", "misinformation"]
articles = collector.collect_and_save(
    queries=queries,
    from_date="2024-01-01",
    to_date="2024-12-31",
    out_path="data/raw/articles.csv",
    page_size=10,
    monthly=True  # Search monthly for better coverage
)
```

#### 2. Data Preprocessing

```python
from src.preprocess import load_raw_articles, build_corpus

df = load_raw_articles("data/raw/articles.csv")
df_processed = build_corpus(df)
df_processed.to_csv("data/processed/corpus.csv", index=False)
```

#### 3. Sentiment Analysis

```python
from src.sentiment import annotate_sentiment

df_with_sentiment = annotate_sentiment(df_processed)
```

#### 4. Ranking

```python
from src.ranking import TfidfRanker

ranker = TfidfRanker()
ranker.fit(df_with_sentiment["text"].tolist())
results = ranker.search("AI deepfake", top_k=10)
```

#### 5. Visualization

```python
from src.visualize import plot_sentiment_over_time

plot_sentiment_over_time(df_with_sentiment)
```

## Features

- **News Collection**: Collect news articles via New York Times Article Search API
- **Monthly Search**: Automatically splits queries by month for better temporal coverage
- **Preprocessing**: URL removal, text normalization
- **Sentiment Analysis**: Sentiment polarity analysis using TextBlob
- **Ranking**: TF-IDF and BM25 algorithms for relevance ranking
- **Visualization**: Time-series visualization of sentiment changes
- **Rate Limiting**: Automatic rate limit management and retry logic

## Configuration

### Environment Variables

`.env` file should contain:

```
NYT_API_KEY=your_api_key_here
```

Get your free API key from [NYT Developer Portal](https://developer.nytimes.com/).

### Settings

In `main.py`, you can adjust:

- `MAX_PAGES`: Maximum pages per query (default: 3 for test mode, use 20-50 for production)
- `page_size_per_query`: Articles per search (default: 10)
- `monthly`: Whether to search monthly (default: True)

## Output Files

Files are automatically saved with timestamps:
- `data/raw/articles_YYYYMMDD_NNN.csv`
- `data/processed/corpus_with_sentiment_YYYYMMDD_NNN.csv`
- `data/figures/sentiment_over_time_YYYYMMDD_NNN.png`

## License

MIT License
