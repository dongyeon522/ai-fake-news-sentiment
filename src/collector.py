# src/collector.py
import requests
import pandas as pd
import json
import os
import time
from datetime import datetime, date
from typing import List, Dict, Optional
from pathlib import Path

class NewsCollector:
    DAILY_LIMIT = 5000  # NYT API free plan daily request limit
    MAX_PAGES = 3  # Test mode (change to 20 or 50 for production)
    
    def __init__(self, api_key: str, request_log_path: str = "data/request_log.json", max_pages: int = None):
        self.api_key = api_key
        self.base_url = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
        self.request_log_path = request_log_path
        self.request_count = self._load_request_count()
        # Use class default if max_pages not specified
        self.max_pages = max_pages if max_pages is not None else self.MAX_PAGES
    
    def _load_request_count(self) -> int:
        """Load saved request count"""
        if not os.path.exists(self.request_log_path):
            return 0
        
        try:
            with open(self.request_log_path, 'r') as f:
                log_data = json.load(f)
                last_date = log_data.get('last_date')
                today = str(date.today())
                
                # Reset if not today's date
                if last_date != today:
                    return 0
                
                return log_data.get('count', 0)
        except (json.JSONDecodeError, KeyError):
            return 0
    
    def _save_request_count(self):
        """Save request count"""
        os.makedirs(os.path.dirname(self.request_log_path), exist_ok=True)
        log_data = {
            'last_date': str(date.today()),
            'count': self.request_count,
            'last_updated': datetime.now().isoformat()
        }
        with open(self.request_log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
    
    def _check_rate_limit(self) -> bool:
        """Check request limit"""
        if self.request_count >= self.DAILY_LIMIT:
            return False
        return True
    
    def _convert_date_format(self, date_str: str) -> str:
        """Convert YYYY-MM-DD format to YYYYMMDD format"""
        return date_str.replace("-", "")
    
    def fetch_articles(self, query: str, from_date: str, to_date: str, page_size: int = 10) -> List[Dict]:
        """Fetch news articles (with rate limit checking)"""
        # Check rate limit
        if not self._check_rate_limit():
            raise Exception(
                f"Daily request limit ({self.DAILY_LIMIT}) reached. "
                f"Please try again tomorrow or use a paid plan."
            )
        
        all_articles = []
        begin_date = self._convert_date_format(from_date)
        end_date = self._convert_date_format(to_date)
        
        # NYT API returns max 10 articles per page, so pagination is needed
        # YAML spec: max 100 pages (1,000 results)
        # Use self.max_pages to distinguish test/production mode
        max_pages = min((page_size // 10) + 1, self.max_pages, 100)  # Apply MAX_PAGES limit
        
        for page in range(max_pages):
            params = {
                "q": query,
                "begin_date": begin_date,
                "end_date": end_date,
                "page": page,
                "api-key": self.api_key,
                "sort": "oldest",  # Changed to oldest to collect evenly across the period
            }
            
            resp = requests.get(self.base_url, params=params)
            
            # Error handling (according to YAML spec error codes)
            if resp.status_code == 400:
                raise Exception(f"Bad request. Check query parameters: {resp.text}")
            elif resp.status_code == 401:
                raise Exception("Authentication failed. Check API key.")
            elif resp.status_code == 429:
                # Retry logic when rate limit exceeded
                # Use API-provided Retry-After if available, otherwise use shorter default
                retry_after = resp.headers.get('Retry-After')
                if retry_after:
                    retry_after = int(retry_after)
                else:
                    # Shorter default wait time (15 seconds instead of 60)
                    retry_after = 15
                
                print(f"      WARNING: Rate limit exceeded. Waiting {retry_after} seconds before retry...")
                print(f"      Progress: {self.request_count}/{self.DAILY_LIMIT} requests used")
                time.sleep(retry_after)
                # Retry
                resp = requests.get(self.base_url, params=params)
                if resp.status_code == 429:
                    # If still rate limited, wait a bit longer
                    retry_after = 30
                    print(f"      Still rate limited. Waiting {retry_after} more seconds...")
                    time.sleep(retry_after)
                    resp = requests.get(self.base_url, params=params)
                    if resp.status_code == 429:
                        raise Exception(f"Rate limit exceeded. Please try again later.")
            elif resp.status_code != 200:
                resp.raise_for_status()
            
            # Add delay between requests (prevent per-minute rate limit)
            if page < max_pages - 1:  # If not last page
                time.sleep(1.0)  # Increased to 1 second to avoid rate limiting
            
            # Progress indicator for long-running operations
            if page == 0:
                print(f"    Processing page {page + 1}/{max_pages}...")
            
            # Increment request count (save in batches for better performance)
            self.request_count += 1
            # Save every 10 requests instead of every request
            if self.request_count % 10 == 0:
                self._save_request_count()
            
            data = resp.json()
            
            # Check response status
            if data.get("status") != "OK":
                raise Exception(f"API response error: {data.get('status')}")
            
            response_data = data.get("response", {})
            docs = response_data.get("docs", [])
            
            # Log meta information (optional)
            meta = response_data.get("meta", {})
            total_hits = meta.get("hits", 0)
            if page == 0 and total_hits > 0:
                print(f"      Found {total_hits} total results (max 10,000 displayed)")
            
            if not docs:
                break  # Stop if no more articles
            
            # Convert NYT API response to standard format
            # According to YAML spec, there is no abstract field, only snippet
            for doc in docs:
                headline = doc.get("headline", {})
                byline = doc.get("byline", {})
                
                # Use snippet for both description and content
                snippet = doc.get("snippet", "")
                
                article = {
                    "title": headline.get("main", ""),
                    "description": snippet,
                    "content": snippet,  # NYT API has no abstract field, so use snippet
                    "url": doc.get("web_url", ""),
                    "published_at": doc.get("pub_date", ""),
                    "author": byline.get("original", "") if byline else "",
                    "source": "The New York Times",
                }
                all_articles.append(article)
            
            # Stop if required number of articles is reached
            if len(all_articles) >= page_size:
                break
        
        return all_articles[:page_size]

    def collect_and_save(self, queries: List[str], from_date: str, to_date: str, out_path: str, page_size: int = None, monthly: bool = True):
        """Collect and save news from multiple queries
        
        Args:
            queries: List of search queries
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            out_path: Output path
            page_size: Number of articles to collect per search (None uses MAX_PAGES * 10)
            monthly: Whether to split queries by month (default: True)
        """
        import pandas as pd
        from datetime import datetime
        
        all_articles = []
        
        # Use MAX_PAGES * 10 if page_size not specified
        if page_size is None:
            page_size = self.max_pages * 10
        
        # Decide whether to split queries by month
        if monthly:
            # Split date range by month
            start_date = pd.to_datetime(from_date)
            end_date = pd.to_datetime(to_date)
            
            # Generate start and end dates for each month
            date_ranges = []
            current = start_date
            while current <= end_date:
                # Last day of current month
                if current.month == 12:
                    month_end = pd.Timestamp(year=current.year, month=12, day=31)
                else:
                    month_end = pd.Timestamp(year=current.year, month=current.month+1, day=1) - pd.Timedelta(days=1)
                
                # Adjust to not exceed end_date
                month_end = min(month_end, end_date)
                
                date_ranges.append((
                    current.strftime('%Y-%m-%d'),
                    month_end.strftime('%Y-%m-%d')
                ))
                
                # Move to next month
                if current.month == 12:
                    current = pd.Timestamp(year=current.year+1, month=1, day=1)
                else:
                    current = pd.Timestamp(year=current.year, month=current.month+1, day=1)
            
            total_searches = len(queries) * len(date_ranges)
            print(f"  Daily request limit: {self.DAILY_LIMIT}")
            print(f"  Current usage: {self.request_count}/{self.DAILY_LIMIT}")
            print(f"  Remaining requests: {self.DAILY_LIMIT - self.request_count}")
            print(f"  Monthly search mode: {len(date_ranges)} months × {len(queries)} queries = {total_searches} total searches")
            
            if self.request_count + total_searches > self.DAILY_LIMIT:
                available = self.DAILY_LIMIT - self.request_count
                print(f"\n  WARNING: May exceed request limit.")
                print(f"  Available requests: {available}, Required: {total_searches}")
                # Only show warning, don't auto-adjust
            
            search_count = 0
            for q_idx, q in enumerate(queries, 1):
                for month_idx, (month_start, month_end) in enumerate(date_ranges, 1):
                    if not self._check_rate_limit():
                        print(f"\n  WARNING: Rate limit reached. {search_count}/{total_searches} searches completed.")
                        break
                    
                    search_count += 1
                    month_label = pd.to_datetime(month_start).strftime('%Y-%m')
                    print(f"\nretrieving {month_label} data... (query: '{q}')", flush=True)
                    print(f"  Progress: {search_count}/{total_searches} searches ({search_count/total_searches*100:.1f}%)", flush=True)
                    print(f"  Request {self.request_count + 1}/{self.DAILY_LIMIT}, max {page_size} articles", flush=True)
                    
                    try:
                        articles = self.fetch_articles(q, month_start, month_end, page_size=page_size)
                        print(f"  Collected {len(articles)} articles", flush=True)
                        
                        for a in articles:
                            all_articles.append({
                                "query": q,
                                "source": a.get("source", "The New York Times"),
                                "author": a.get("author", ""),
                                "title": a.get("title", ""),
                                "description": a.get("description", ""),
                                "content": a.get("content", ""),
                                "url": a.get("url", ""),
                                "published_at": a.get("published_at", ""),
                            })
                    except Exception as e:
                        print(f"    Error occurred: {e}")
                        # Continue even if some monthly searches fail
                        continue
                
                if not self._check_rate_limit():
                    break
        else:
            # Non-monthly search (original method)
            remaining_queries = len(queries)
            
            print(f"  Daily request limit: {self.DAILY_LIMIT}")
            print(f"  Current usage: {self.request_count}/{self.DAILY_LIMIT}")
            print(f"  Remaining requests: {self.DAILY_LIMIT - self.request_count}")
            print(f"  Number of queries to collect: {remaining_queries}")
            
            if self.request_count + remaining_queries > self.DAILY_LIMIT:
                available = self.DAILY_LIMIT - self.request_count
                print(f"\n  WARNING: May exceed request limit.")
                print(f"  Available requests: {available}, Required: {remaining_queries}")
                response = input(f"  Process only {available} queries? (y/n): ")
                if response.lower() == 'y':
                    queries = queries[:available]
                    remaining_queries = available
            
            for i, q in enumerate(queries, 1):
                if not self._check_rate_limit():
                    print(f"\n  WARNING: Rate limit reached. {i-1}/{len(queries)} queries completed.")
                    break
                
                print(f"  [{i}/{len(queries)}] Processing query '{q}'... (Request {self.request_count + 1}/{self.DAILY_LIMIT}, max {page_size} articles)")
                try:
                    articles = self.fetch_articles(q, from_date, to_date, page_size=page_size)
                    print(f"    Collected {len(articles)} articles")
                    
                    for a in articles:
                        all_articles.append({
                            "query": q,
                            "source": a.get("source", "The New York Times"),
                            "author": a.get("author", ""),
                            "title": a.get("title", ""),
                            "description": a.get("description", ""),
                            "content": a.get("content", ""),
                            "url": a.get("url", ""),
                            "published_at": a.get("published_at", ""),
                        })
                except Exception as e:
                    print(f"    ✗ Error occurred: {e}")
                    raise
        
        if len(all_articles) == 0:
            raise Exception("No articles collected.")
        
        # Save final request count
        self._save_request_count()
        
        df = pd.DataFrame(all_articles)
        df.to_csv(out_path, index=False)
        print(f"\n  Collected {len(df)} articles in total.")
        print(f"  Remaining requests: {self.DAILY_LIMIT - self.request_count}/{self.DAILY_LIMIT}")
        return df
    
    def get_remaining_requests(self) -> int:
        """Return remaining request count"""
        return max(0, self.DAILY_LIMIT - self.request_count)
