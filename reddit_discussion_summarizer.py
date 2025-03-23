"""
This agent targets discussions on Reddit. It fetches posts (titles) from a 
specified subreddit (and an optional search query), aggregates the content, 
and then leverages Hugging Face pipelines to generate a summary and perform 
sentiment analysis. This agent demonstrates web scraping, NLP pipeline 
integration, and social media analysis.
"""

import requests
from transformers import pipeline
import torch

def fetch_reddit_posts(subreddit, query=None):
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
        posts = [child["data"]["title"] for child in data["data"]["children"]]
        return posts
    except Exception as e:
        print(f"Error fetching Reddit posts: {e}")
        return []

def main():
    subreddit = input("Enter the subreddit (e.g., news, technology): ").strip()
    query = input("Enter a search query (optional): ").strip() or None
    posts = fetch_reddit_posts(subreddit, query)
    
    if not posts:
        print("No posts found. Please check the subreddit or query.")
        return
    
    print(f"\nFetched {len(posts)} posts from r/{subreddit}.")
    aggregated_text = "\n".join(posts)
    print("\nAggregated Post Titles (first 500 characters):")
    print(aggregated_text[:500] + "...")
    
    # Configure device for Hugging Face pipelines
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS acceleration")
    
    # Initialize Hugging Face pipelines with device settings
    summarizer = pipeline(
        "summarization", 
        model="facebook/bart-large-cnn",
        device=0 if device != "cpu" else -1
    )
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        device=0 if device != "cpu" else -1
    )
    
    # For summarization, limit text length if necessary
    words = aggregated_text.split()
    text_to_summarize = aggregated_text if len(words) < 500 else " ".join(words[:500])
    summary = summarizer(text_to_summarize, max_length=130, min_length=30, do_sample=False)
    sentiment = sentiment_analyzer(aggregated_text[:512])
    
    print("\nSummary of Reddit Discussion:")
    print(summary[0]['summary_text'])
    print("\nSentiment Analysis:")
    for result in sentiment:
        print(f"Label: {result['label']}, Score: {result['score']:.4f}")

if __name__ == "__main__":
    main()