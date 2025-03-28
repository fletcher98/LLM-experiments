"""
This agent targets discussions on Reddit. It fetches posts (titles) from a 
specified subreddit (and an optional search query), aggregates the content, 
and then leverages Hugging Face pipelines to generate a summary and perform 
sentiment analysis. This agent demonstrates web scraping, NLP pipeline 
integration, and social media analysis.
"""

import argparse
import logging
import requests
from transformers import pipeline
import torch
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def fetch_reddit_posts(subreddit: str, query: Optional[str] = None) -> List[str]:
    """
    Fetches post titles from a specified subreddit, optionally filtered by a search query.
    
    Args:
        subreddit (str): The target subreddit.
        query (Optional[str]): An optional search query.
    
    Returns:
        List[str]: A list of post titles.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    if query:
        url = f"https://www.reddit.com/r/{subreddit}/search.json"
        params = {"q": query, "restrict_sr": "on", "sort": "relevance", "limit": 50}
    else:
        url = f"https://www.reddit.com/r/{subreddit}/.json"
        params = {"limit": 50}
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        posts = [child["data"]["title"] for child in data.get("data", {}).get("children", [])]
        logging.info(f"Fetched {len(posts)} posts from r/{subreddit}.")
        return posts
    except Exception as e:
        logging.error(f"Error fetching Reddit posts: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Reddit Discussion Summarizer and Sentiment Analyzer."
    )
    parser.add_argument("--subreddit", type=str, help="Subreddit to fetch posts from.")
    parser.add_argument("--query", type=str, help="Optional search query.")
    args = parser.parse_args()
    
    subreddit = args.subreddit or input("Enter the subreddit (e.g., news, technology): ").strip()
    query = args.query if args.query is not None else input("Enter a search query (optional): ").strip() or None

    posts = fetch_reddit_posts(subreddit, query)
    
    if not posts:
        logging.error("No posts found. Please check the subreddit or query.")
        return
    
    aggregated_text = "\n".join(posts)
    logging.info(f"Aggregated text length: {len(aggregated_text)} characters.")
    
    # Show sample of aggregated text (first 500 characters)
    print("\nAggregated Post Titles (first 500 characters):")
    print(aggregated_text[:500] + "...")
    
    # Device selection: prefer CUDA over MPS, then CPU
    if torch.cuda.is_available():
        device_name = "cuda"
    elif torch.backends.mps.is_available():
        device_name = "mps"
    else:
        device_name = "cpu"
    logging.info(f"Using device: {device_name}")
    
    # Initialize Hugging Face pipelines with appropriate device settings
    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        device=0 if device_name != "cpu" else -1
    )
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        device=0 if device_name != "cpu" else -1
    )
    
    # For summarization, limit text length if necessary
    words = aggregated_text.split()
    if len(words) > 500:
        text_to_summarize = " ".join(words[:500])
        logging.info("Aggregated text truncated for summarization.")
    else:
        text_to_summarize = aggregated_text
    
    try:
        summary_output = summarizer(text_to_summarize, max_length=130, min_length=30, do_sample=False)
        summary_text = summary_output[0]['summary_text']
    except Exception as e:
        logging.error(f"Error during summarization: {e}")
        summary_text = "Summarization failed."
    
    try:
        # Sentiment analysis on a segment of aggregated text (limited to first 512 characters)
        sentiment_results = sentiment_analyzer(aggregated_text[:512])
    except Exception as e:
        logging.error(f"Error during sentiment analysis: {e}")
        sentiment_results = []
    
    print("\nSummary of Reddit Discussion:")
    print(summary_text)
    print("\nSentiment Analysis:")
    for result in sentiment_results:
        print(f"Label: {result['label']}, Score: {result['score']:.4f}")


if __name__ == "__main__":
    main()