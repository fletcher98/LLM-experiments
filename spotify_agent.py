"""
Spotify agent that uses Spotify’s Web API (via the spotipy library) to fetch 
information about an artist, specifically their top tracks and some associated metadata. 
The agent then aggregates key details (such as track names and popularity) and feeds this 
summary to a Hugging Face summarization pipeline (using a model like BART) to generate a 
concise overview of the artist’s musical style. This project demonstrates integration 
with external APIs, data aggregation, and LLM-powered summarization.
"""

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from transformers import pipeline
import torch

def main():
    # Spotify API credentials
    client_id = 
    client_secret = 
    
    # Set up Spotify client using client credentials flow
    client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    
    # Prompt user for an artist name
    artist_name = input("Enter an artist name: ").strip()
    
    # Search for the artist and retrieve the top result
    results = sp.search(q=artist_name, type='artist', limit=1)
    if not results['artists']['items']:
        print("Artist not found. Please try another name.")
        return
    
    artist = results['artists']['items'][0]
    print(f"Found artist: {artist['name']}")
    artist_id = artist['id']
    
    # Retrieve the artist's top tracks in the US market
    top_tracks = sp.artist_top_tracks(artist_id, country='US')
    tracks = top_tracks['tracks']
    if not tracks:
        print("No top tracks found for this artist.")
        return

    print("\nTop Tracks:")
    for t in tracks:
        print("-", t['name'])
    
    # Aggregate track details for summarization
    track_details = ""
    for t in tracks:
        track_details += f"Track: {t['name']}. Popularity: {t['popularity']}. "
    
    # Set up device for Hugging Face pipelines
    device = "cpu"
    if torch.cuda.is_available():
        device = 0  # Use the first CUDA device
    
    # Initialize the summarization pipeline (using a model like BART)
    summarizer = pipeline(
        "summarization", 
        model="facebook/bart-large-cnn",
        device=device
    )
    
    # Summarize the aggregated track details to provide a concise overview of the artist's style
    summary = summarizer(track_details, max_length=60, min_length=30, do_sample=False)
    
    print("\nArtist Summary:")
    print(summary[0]['summary_text'])

if __name__ == "__main__":
    main()