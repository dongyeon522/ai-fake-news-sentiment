# AI Fake News Sentiment Tracking and Analysis

## Documentation Note
This README serves as the official documentation required for the software code submission.  
It includes:
- **How to use the software** (installation, configuration, and execution)
- **How the software is implemented** (module-level explanations and processing pipeline)

## Overview
This project analyzes public perception of AI-generated fake news, deepfakes, and misinformation by collecting news articles and evaluating their sentiment over time.  
The system integrates:

- Article retrieval (via the New York Times Article Search API)
- Text preprocessing and corpus construction
- Sentiment analysis using **TextBlob** and **RoBERTa**
- Retrieval ranking using TF-IDF and BM25
- Sentiment trend visualization across months

By comparing lightweight lexicon-based models with transformer-based sentiment classifiers, the system reveals how media framing of AI-related misinformation differs depending on model sensitivity and context awareness.

### Automated News Retrieval
- Uses the NYT Article Search API  
- Runs monthly queries for four topics:
  - “artificial intelligence”
  - “deepfake”
  - “fake news”
  - “misinformation”
- Automatically handles rate limits and retries  
- Produces raw CSV files under `data/raw/`

### Text Preprocessing
- Concatenates `title + description + content` into a clean text block  
- Normalizes whitespace  
- Produces processed datasets with unified text fields

### Sentiment Analysis
Supports two models:

| Model | Type | Behavior |
|-------|------|----------|
| **TextBlob** | Lexicon-based | Conservative, near-neutral scoring |
| **RoBERTa** | Transformer-based | Captures nuanced, contextual negativity |

### Visualization
- Generates time-series monthly sentiment plots  
- Saves figures to `data/figures/`

### Ranking & Evaluation
- TF-IDF and BM25 ranking modules  
- Evaluation utilities for examining sentiment distributions  

---
## Installation

1. Clone the repository:
```bash
git clone https://github.com/dongyeon522/ai-fake-news-sentiment
cd ai-fake-news-sentiment
```

2. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note**: For RoBERTa sentiment analysis, additional packages (`transformers` and `torch`) are included in `requirements.txt`. These will be installed automatically, but note that they require significant disk space (~1GB+).

4. Set up environment variables:
```bash
cp .env.example .env
```
5. Run the complete pipeline:

```bash
python3 main.py
```

Or use the convenience script:

```bash
./run.sh
```
---
## Implementation Details

The source code is organized into modular components:

### `collector.py` — Article Retrieval
- Interfaces with the NYT Article Search API  
- Performs month-by-month queries  
- Ensures full 2024 coverage across four keywords  
- Handles rate limits and retry logic  
- Saves raw article data to CSV  

### `preprocess.py` — Text Cleaning
- Merges `title`, `description`, and `content` fields  
- Fills missing values with empty strings  
- Produces a unified `text` field for each article  

### `sentiment.py` — Sentiment Modeling
- **TextBlob:** lightweight lexicon-based polarity scoring  
- **RoBERTa:** transformer-based sentiment classifier  
- Adds sentiment score to each processed article  

### `ranking.py` — TF-IDF & BM25 Ranking
- Builds TF-IDF and BM25 vectorizers  
- Enables retrieval and ranking experiments  

### `evaluate.py` — Sentiment Statistics
- Computes positive, neutral, and negative proportions  
- Supports custom sentiment threshold configurations  

### `visualize.py` — Plotting
- Generates monthly sentiment line charts  
- Supports visualization for both sentiment models  

### `main.py` — Orchestration
- Runs the full retrieval → preprocessing → sentiment → evaluation → visualization pipeline  
- Allows selection of the sentiment model at runtime  
- Produces final processed CSVs and figures  

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
- `TEST_MODE`: Set to `False` for full year search (default: `False`)

### Sentiment Analysis Configuration

**Method Selection**: When running `main.py`, you'll be prompted to choose between TextBlob and RoBERTa.

**Threshold Configuration**: In `src/evaluate.py`, the `evaluate_sentiment_distribution()` function accepts threshold parameters:

```python
from src.evaluate import evaluate_sentiment_distribution

# Default thresholds (pos_th=0.1, neg_th=-0.1)
stats = evaluate_sentiment_distribution(df)

# Custom thresholds
stats = evaluate_sentiment_distribution(df, pos_th=0.2, neg_th=-0.2)
```
---
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
├── presentation/     # Project presentation materials
│   ├── final report.pptx  # Project report presentation
│   └── final report.mp4   # Video demonstration
├── main.py           # Main execution script
└── requirements.txt
```





This will:
1. Collect news articles from NYT API (monthly search mode)
2. Preprocess the data
3. **Prompt you to select sentiment analysis method** (TextBlob or RoBERTa)
4. Perform sentiment analysis
5. Evaluate results
6. Run ranking experiments
7. Generate visualizations

**Note**: When running, you'll be prompted to choose between:
- **TextBlob**: Faster, rule-based sentiment analysis
- **RoBERTa**: Slower but more accurate deep learning-based sentiment analysis

## Features

- **News Collection**: Collect news articles via New York Times Article Search API
- **Monthly Search**: Automatically splits queries by month for better temporal coverage
- **Preprocessing**: URL removal, text normalization
- **Sentiment Analysis**: 
  - **TextBlob**: Fast, rule-based sentiment analysis
  - **RoBERTa**: Deep learning-based sentiment analysis (more accurate)
- **Ranking**: TF-IDF and BM25 algorithms for relevance ranking
- **Evaluation**: Comprehensive sentiment distribution statistics with configurable thresholds
- **Visualization**: Time-series visualization of sentiment changes
- **Rate Limiting**: Automatic rate limit management and retry logic (optimized wait times)
- **Progress Tracking**: Real-time progress display during data collection



## Output Files

Files are automatically saved with timestamps:
- `data/raw/articles_YYYYMMDD_NNN.csv` - Raw article data from NYT API
- `data/processed/corpus_with_sentiment_YYYYMMDD_NNN.csv` - Processed data with sentiment scores
- `data/figures/sentiment_over_time_YYYYMMDD_NNN.png` - Time-series visualization

## Example Output

After running the pipeline, you'll see sentiment statistics like:

```
[4/6] Evaluating sentiment results...

Sentiment distribution statistics:
  Mean sentiment: 0.045
  Standard deviation: 0.227
  Positive ratio: 32.39%
  Negative ratio: 16.52%
  Neutral ratio: 51.09%
```

### Generated Files

The pipeline generates three output files:

1. **Raw Data CSV**: `data/raw/articles_YYYYMMDD_NNN.csv`
   - Contains raw article data collected from NYT API
   - Columns: query, source, author, title, description, content, url, published_at

2. **Processed Data CSV**: `data/processed/corpus_with_sentiment_YYYYMMDD_NNN.csv`
   - Contains preprocessed text and sentiment scores
   - Columns: query, source, author, title, description, content, url, published_at, text, sentiment

3. **Visualization PNG**: `data/figures/sentiment_over_time_YYYYMMDD_NNN.png`
   - Time-series plot showing sentiment trends over time
   - Example: `sentiment_over_time_20251205_002.png`
   
   ![Sentiment Over Time](data/figures/sentiment_over_time_20251205_002.png)

## Performance Notes

- **TextBlob**: Fast execution, suitable for large datasets
- **RoBERTa**: Slower but more accurate. First run downloads ~500MB model (cached for subsequent runs)
- **Rate Limiting**: Optimized retry logic with 15-second default wait time (reduced from 60 seconds)
- **Progress Display**: Real-time progress updates during data collection

---
# Final Report

This is a final report and the same content can be found in Textdata with following url:
https://textdata.org/submissions/68eb8d21d79b5a018fcc4039

#project #report

**Overview**

This project aims to analyze how the public perceives AI-generated fake news, deepfakes, and misinformation by collecting related news articles and examining their sentiment. By ranking articles with TF-IDF and BM25 and applying sentiment analysis models such as TextBlob or RoBERTa, the system approximates public reactions reflected in media tone. The goal is to provide a simple, unified workflow that retrieves relevant articles, measures sentiment trends over time, and evaluates retrieval performance using standard IR metrics.

**Progress Made**

Project code: https://github.com/dongyeon522/ai-fake-news-sentiment
 
Presentation video: https://mediaspace.illinois.edu/media/t/1_e137pu64

**(a)	Initial Retrieval and Collection Pipeline**

- Implemented a month-by-month retrieval strategy for the full year of 2024.
- For each month, the system executed four distinct search queries targeting AI misinformation topics:

1.	"artificial intelligence"
2.	"deepfake"
3.	"fake news"
4.	"misinformation"

- For every query–month pair, the system requested up to 10 articles (NYT API page limit), resulting in:
    - 12 months × 4 queries = 48 total API calls
    - Up to 480 articles possible, with 459 successfully collected after removing duplicates
    - Monthly splitting ensured that all months of 2024 were represented, allowing a balanced temporal analysis.

**(b)	Text Preprocessing and Corpus Construction**

- Implemented a preprocessing module that concatenates each article’s **title, description, and content** into a single `text` field.
- Filled missing fields with empty strings and normalized basic whitespace so that every row has a clean, model-ready text block.
- Starting from the raw NYT API export (`data/raw/articles_YYYYMMDD_XXX.csv`), the pipeline produces a processed file  
  `data/processed/corpus_with_sentiment_YYYYMMDD_XXX.csv` that adds the unified `text` column and a `sentiment` score for each article.

**(c)	Sentiment Analysis**

Two sentiment models were applied:

1. **TextBlob** (lexicon-based polarity model)  
2. **RoBERTa** (transformer-based classifier)

Their monthly sentiment trends show clear differences in sensitivity and range.

(1) TextBlob Sentiment Trend

![TextBlob result](data/figures/sentiment_over_time_20251205_002.png)
https://github.com/dongyeon522/ai-fake-news-sentiment/blob/main/data/figures/sentiment_over_time_20251205_002.png

**Observations:**
- Sentiment remains close to **neutral**, with only minor fluctuations.
- This behavior reflects TextBlob's tendency to compress polarity for formal news text.
- Positive/negative classification used the common threshold **±0.1**.

(2) RoBERTa Sentiment Trend

![RoBERTa result](data/figures/sentiment_over_time_20251205_004.png)
https://github.com/dongyeon522/ai-fake-news-sentiment/blob/main/data/figures/sentiment_over_time_20251205_004.png

**Observations:**
- RoBERTa produces **stronger negative sentiment** across the year.
- Negative dips occur around:
  - **February (–0.25)**
  - **April–June (–0.18 to –0.22)**
  - **Late fall (–0.20 to –0.36)**
- This model captures nuance such as concern, risk framing, or negative social impact.
- Sharp contrast with TextBlob reveals the importance of model selection.


**Interpretation**
- **TextBlob** indicates that AI misinformation articles appear neutral in tone overall.
- **RoBERTa**, however, reveals **consistently negative framing**, with notable dips aligning with:
  - regulatory debates,
  - high-profile misuse incidents,
  - or controversies involving major AI firms.

The divergence highlights how transformer-based models better detect subtle negativity present in journalistic descriptions of AI risk, misinformation, and deepfake threats.

**Challenges / Issues Encountered**

**(a)	API Constraints**
- NYT API provides only 10 items per page.  
- Used monthly splitting to avoid missing data.
- Some months contain inherently fewer AI-misinformation stories.

(b)	Sentiment Model Limitations
- TextBlob polarity compresses toward zero for formal writing.
- RoBERTa is more expressive but computationally more costly.

**Conclusion**

- The project successfully built an end-to-end pipeline that retrieves AI-misinformation news, preprocesses text, and analyzes sentiment over time.
- TextBlob showed sentiment clustered near neutrality, reflecting its limitations in detecting nuanced tone in formal news writing.
- RoBERTa revealed consistently negative sentiment throughout 2024, capturing concern-driven or risk-oriented framing that lexicon models fail to detect.
- The contrast between the two models highlights the importance of model selection for analyzing subtle or context-dependent sentiment in news media.
- Despite API rate limits and variability in model behavior, the system provides a practical foundation for monitoring public sentiment toward AI misinformation and can be extended with more robust retrieval evaluation and advanced semantic models.


## License
This project is released for academic coursework submission and is not licensed for commercial use.