"""
This project is a web agent that analyzes news articles by combining web scraping 
with natural language processing techniques. It retrieves an article from a 
user-provided URL, extracts its text using BeautifulSoup, and then uses Hugging Face 
pipelines to generate a concise summary and perform sentiment analysis. Additionally, 
it conducts a basic bias check by counting gendered words to assess potential skew in 
the content. This integration of web data extraction, summarization, sentiment evaluation, 
and bias detection showcases a comprehensive skill set that aligns well with research and 
development needs in large language model applications.
"""

import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import re
from collections import Counter
import torch

# Function to fetch and extract text from a news article URL
def fetch_article_text(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        # Attempt to extract text from common article tags
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])
        return text.strip()
    except Exception as e:
        print(f"Error fetching article: {e}")
        return ""

# Simple bias analysis function based on counting gendered/stereotypical words
def analyze_bias(text):
    # Define some simple bias indicators
    gendered_words = {
        "male": ["he", "him", "his", "man", "men", "male"],
        "female": ["she", "her", "hers", "woman", "women", "female"]
    }
    tokens = re.findall(r'\w+', text.lower())
    counts = {"male": 0, "female": 0}
    token_counts = Counter(tokens)
    for gender, words in gendered_words.items():
        for word in words:
            counts[gender] += token_counts.get(word, 0)
    return counts

def main():
    # Prompt user for a news article URL
    url = input("Enter the URL of a news article: ").strip()
    article_text = fetch_article_text(url)
    
    if not article_text:
        print("Failed to retrieve article text. Please check the URL.")
        return
    
    print("\nExtracted Article Text (first 500 characters):")
    print(article_text[:500] + "...\n")
    
    # Configure device for model inference
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Metal Performance Shaders) acceleration")
    
    # Initialize Hugging Face pipelines with device and specific models
    print("Analyzing the article...")
    summarizer = pipeline(
        "summarization", 
        model="facebook/bart-large-cnn",
        device=device
    )
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        revision="714eb0f",
        device=device
    )
    
    # Summarize the article (truncate if too long)
    # The summarization pipeline may have a max token limit; we slice the text if needed.
    if len(article_text.split()) > 500:
        summary_input = " ".join(article_text.split()[:500])
    else:
        summary_input = article_text
    
    summary = summarizer(summary_input, max_length=130, min_length=30, do_sample=False)
    sentiment = sentiment_analyzer(article_text[:512])  # analyzing first 512 tokens
    
    # Analyze bias via simple keyword counting
    bias_counts = analyze_bias(article_text)
    
    # Display the results
    print("Summary of the Article:")
    print(summary[0]['summary_text'])
    print("\nSentiment Analysis:")
    for result in sentiment:
        print(f"Label: {result['label']}, Score: {result['score']:.4f}")
    print("\nBias Indicator Counts (based on simple keyword matching):")
    print(f"Male-related words: {bias_counts['male']}")
    print(f"Female-related words: {bias_counts['female']}")

if __name__ == "__main__":
    main()
