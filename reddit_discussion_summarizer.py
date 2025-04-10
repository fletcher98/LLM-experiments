"""
This agent targets discussions on Reddit. It fetches posts and comments from a 
specified subreddit (and an optional search query), aggregates the content, 
and then leverages Hugging Face pipelines to generate a summary, perform 
sentiment analysis, and extract key topics. This agent demonstrates web scraping, 
NLP pipeline integration, and social media analysis.
"""

import argparse
import logging
import requests
import time
import json
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from transformers import pipeline
import torch
from typing import List, Optional, Dict, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class RedditAPI:
    """Handles Reddit API interactions with rate limiting."""
    
    def __init__(self):
        self.headers = {"User-Agent": "Mozilla/5.0"}
        self.last_request_time = 0
        self.min_request_interval = 2  # seconds between requests
    
    def _rate_limit(self):
        """Implements rate limiting for API requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()

    def fetch_reddit_posts(
        self, 
        subreddit: str, 
        query: Optional[str] = None,
        sort: str = "hot",
        time_filter: str = "month",
        limit: int = 50
    ) -> List[Dict]:
        """
        Fetches posts from a specified subreddit with various filtering options.
        
        Args:
            subreddit (str): The target subreddit
            query (Optional[str]): Optional search query
            sort (str): Sorting method ('hot', 'new', 'top', 'rising')
            time_filter (str): Time filter for results ('hour', 'day', 'week', 'month', 'year', 'all')
            limit (int): Maximum number of posts to fetch
            
        Returns:
            List[Dict]: List of post data including titles, content, and metadata
        """
        self._rate_limit()
        
        if query:
            url = f"https://www.reddit.com/r/{subreddit}/search.json"
            params = {
                "q": query,
                "restrict_sr": "on",
                "sort": sort,
                "t": time_filter,
                "limit": limit
            }
        else:
            url = f"https://www.reddit.com/r/{subreddit}/{sort}.json"
            params = {"t": time_filter, "limit": limit}
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            posts = []
            for child in data.get("data", {}).get("children", []):
                post_data = child["data"]
                posts.append({
                    "title": post_data["title"],
                    "selftext": post_data.get("selftext", ""),
                    "score": post_data["score"],
                    "url": f"https://reddit.com{post_data['permalink']}",
                    "created_utc": post_data["created_utc"],
                    "num_comments": post_data["num_comments"]
                })
            
            logging.info(f"Fetched {len(posts)} posts from r/{subreddit}.")
            return posts
            
        except Exception as e:
            logging.error(f"Error fetching Reddit posts: {e}")
            return []

    def fetch_comments(self, post_url: str, limit: int = 10) -> List[str]:
        """Fetches top comments from a Reddit post."""
        self._rate_limit()
        
        try:
            url = f"{post_url}.json"
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if len(data) < 2:
                return []
                
            comments = []
            comment_data = data[1]["data"]["children"]
            
            for comment in comment_data[:limit]:
                if comment["kind"] == "t1":  # Regular comment
                    comments.append(comment["data"]["body"])
                    
            return comments
            
        except Exception as e:
            logging.error(f"Error fetching comments: {e}")
            return []

def extract_topics(text: str, topic_extractor) -> List[str]:
    """Extracts key topics from text using zero-shot classification."""
    try:
        candidate_topics = [
            "technology", "politics", "science", "entertainment",
            "business", "sports", "health", "education", "environment",
            "society", "culture", "economy"
        ]
        
        results = topic_extractor(text, candidate_topics, multi_label=True)
        relevant_topics = [label for label, score in zip(results["labels"], results["scores"]) if score > 0.3]
        return relevant_topics
        
    except Exception as e:
        logging.error(f"Error extracting topics: {e}")
        return []

def save_results(subreddit: str, data: Dict, output_dir: str = "results"):
    """Saves analysis results to a JSON file."""
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_dir / f"reddit_analysis_{subreddit}_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        logging.info(f"Results saved to {filename}")
        
    except Exception as e:
        logging.error(f"Error saving results: {e}")

def aggregate_content(posts: List[Dict], reddit_api: RedditAPI) -> Tuple[str, str]:
    """Aggregates post content and comments."""
    titles = []
    full_content = []
    
    for post in tqdm(posts, desc="Fetching comments"):
        titles.append(post["title"])
        
        content_parts = [post["title"]]
        if post["selftext"]:
            content_parts.append(post["selftext"])
            
        # Fetch and add top comments
        comments = reddit_api.fetch_comments(post["url"])
        content_parts.extend(comments)
        
        full_content.append("\n".join(content_parts))
    
    return "\n".join(titles), "\n\n".join(full_content)

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Reddit Discussion Summarizer and Analyzer."
    )
    parser.add_argument("--subreddit", type=str, help="Subreddit to fetch posts from")
    parser.add_argument("--query", type=str, help="Optional search query")
    parser.add_argument("--sort", type=str, choices=["hot", "new", "top", "rising"], 
                       default="hot", help="Sort method for posts")
    parser.add_argument("--time", type=str, 
                       choices=["hour", "day", "week", "month", "year", "all"],
                       default="month", help="Time filter for posts")
    parser.add_argument("--limit", type=int, default=50,
                       help="Maximum number of posts to fetch")
    parser.add_argument("--save", action="store_true",
                       help="Save results to file")
    args = parser.parse_args()
    
    # Get input if not provided as arguments
    subreddit = args.subreddit or input("Enter the subreddit (e.g., news, technology): ").strip()
    query = args.query if args.query is not None else input("Enter a search query (optional): ").strip() or None
    
    # Initialize Reddit API handler
    reddit_api = RedditAPI()
    
    # Fetch posts with progress indication
    posts = reddit_api.fetch_reddit_posts(
        subreddit=subreddit,
        query=query,
        sort=args.sort,
        time_filter=args.time,
        limit=args.limit
    )
    
    if not posts:
        logging.error("No posts found. Please check the subreddit or query.")
        return
    
    # Aggregate content with progress bar
    titles_text, full_content = aggregate_content(posts, reddit_api)
    logging.info(f"Aggregated content length: {len(full_content)} characters")
    
    # Device selection: prefer CUDA over MPS, then CPU
    if torch.cuda.is_available():
        device_name = "cuda"
    elif torch.backends.mps.is_available():
        device_name = "mps"
    else:
        device_name = "cpu"
    logging.info(f"Using device: {device_name}")
    
    # Initialize models with progress indication
    print("\nInitializing models...")
    models = {}
    for name, model_info in tqdm([
        ("summarizer", ("summarization", "facebook/bart-large-cnn")),
        ("sentiment", ("sentiment-analysis", "distilbert/distilbert-base-uncased-finetuned-sst-2-english")),
        ("topics", ("zero-shot-classification", "facebook/bart-large-mnli"))
    ], desc="Loading models"):
        task, model = model_info
        models[name] = pipeline(
            task,
            model=model,
            device=0 if device_name != "cpu" else -1
        )
    
    # Process content in chunks if necessary
    max_chunk_size = 500  # words
    words = full_content.split()
    chunks = [" ".join(words[i:i + max_chunk_size]) 
             for i in range(0, len(words), max_chunk_size)]
    
    results = {
        "subreddit": subreddit,
        "query": query,
        "timestamp": datetime.now().isoformat(),
        "metadata": {
            "post_count": len(posts),
            "sort_method": args.sort,
            "time_filter": args.time
        }
    }
    
    # Generate summary
    try:
        summaries = []
        for chunk in tqdm(chunks[:3], desc="Generating summary"):  # Process up to 3 chunks
            summary_output = models["summarizer"](chunk, max_length=130, min_length=30, do_sample=False)
            summaries.append(summary_output[0]['summary_text'])
        results["summary"] = " ".join(summaries)
    except Exception as e:
        logging.error(f"Error during summarization: {e}")
        results["summary"] = "Summarization failed."
    
    # Perform sentiment analysis
    try:
        sentiment_results = models["sentiment"](titles_text[:512])
        results["sentiment"] = sentiment_results[0]
    except Exception as e:
        logging.error(f"Error during sentiment analysis: {e}")
        results["sentiment"] = {"label": "ERROR", "score": 0.0}
    
    # Extract topics
    results["topics"] = extract_topics(titles_text, models["topics"])
    
    # Display results
    print("\nSummary of Reddit Discussion:")
    print(results["summary"])
    
    print("\nSentiment Analysis:")
    print(f"Label: {results['sentiment']['label']}, Score: {results['sentiment']['score']:.4f}")
    
    print("\nKey Topics:")
    print(", ".join(results["topics"]) if results["topics"] else "No clear topics identified")
    
    # Save results if requested
    if args.save:
        save_results(subreddit, results)

if __name__ == "__main__":
    main()