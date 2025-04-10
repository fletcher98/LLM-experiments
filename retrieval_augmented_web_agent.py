"""
Enhanced web retrieval agent with improved scraping, caching, chunking, and content processing.
Features include concurrent fetching, content filtering, source tracking, and smart retrieval.
"""

import os
import time
import json
import hashlib
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
from functools import lru_cache

import torch
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')
import requests
from bs4 import BeautifulSoup, Comment
from googlesearch import search
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from newspaper3k import Article
from readability import Document
import html2text
from ratelimit import limits, sleep_and_retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class Config:
    """Centralized configuration management."""
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "google/gemma-2b-it")
    CACHE_DIR: str = os.getenv("CACHE_DIR", "web_cache")
    CACHE_EXPIRY_DAYS: int = 1
    MAX_WORKERS: int = 5
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    MAX_TOKENS: int = 1024
    MIN_CONTENT_LENGTH: int = 100
    REQUESTS_PER_SECOND: int = 2

@dataclass
class WebChunk:
    """Represents a chunk of web content with metadata."""
    text: str
    url: str
    title: str
    timestamp: datetime
    chunk_index: int
    
    def get_citation(self) -> str:
        """Generate a citation for the chunk."""
        domain = urlparse(self.url).netloc
        return f"{self.title} ({domain}, {self.timestamp.strftime('%Y-%m-%d')})"

class WebCache:
    """Manages caching of web content and embeddings."""
    def __init__(self, cache_dir: str = Config.CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self._clean_old_cache()
    
    def _clean_old_cache(self):
        """Remove cached files older than expiry period."""
        expiry = datetime.now() - timedelta(days=Config.CACHE_EXPIRY_DAYS)
        for cache_file in self.cache_dir.glob("*.cache"):
            if datetime.fromtimestamp(cache_file.stat().st_mtime) < expiry:
                cache_file.unlink()
    
    def get_cache_path(self, key: str) -> Path:
        """Get cache file path for a key."""
        return self.cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.cache"
    
    def get(self, key: str) -> Optional[Dict]:
        """Retrieve cached data."""
        cache_path = self.get_cache_path(key)
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                return json.load(f)
        return None
    
    def set(self, key: str, data: Dict):
        """Cache data."""
        cache_path = self.get_cache_path(key)
        with open(cache_path, 'w') as f:
            json.dump(data, f)

class ContentCleaner:
    """Cleans and processes web content."""
    def __init__(self):
        self.h2t = html2text.HTML2Text()
        self.h2t.ignore_links = True
        self.h2t.ignore_images = True
    
    def clean_html(self, html: str) -> str:
        """Clean HTML content using multiple methods."""
        # Parse with readability
        doc = Document(html)
        main_content = doc.summary()
        
        # Convert to markdown/text
        text = self.h2t.handle(main_content)
        
        # Clean up text
        lines = text.split('\n')
        lines = [line.strip() for line in lines if line.strip()]
        return ' '.join(lines)
    
    def extract_article_content(self, url: str, html: str) -> Tuple[str, str]:
        """Extract article content using newspaper3k."""
        article = Article(url)
        article.download(input_html=html)
        article.parse()
        
        if not article.text:
            # Fallback to basic cleaning if newspaper3k fails
            return article.title or "", self.clean_html(html)
        
        return article.title, article.text

@sleep_and_retry
@limits(calls=Config.REQUESTS_PER_SECOND, period=1)
def fetch_url(url: str) -> Optional[Tuple[str, str]]:
    """Fetch URL content with rate limiting and error handling."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return url, response.text
    except Exception as e:
        logger.error(f"Error fetching {url}: {e}")
        return None

def chunk_text(text: str, chunk_size: int = Config.CHUNK_SIZE, 
               overlap: int = Config.CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk.split()) >= Config.MIN_CONTENT_LENGTH:
            chunks.append(chunk)
    return chunks

class WebRetriever:
    """Manages web content retrieval and processing."""
    def __init__(self):
        self.cache = WebCache()
        self.cleaner = ContentCleaner()
        
    def get_web_results(self, query: str, num_results: int = 5) -> List[WebChunk]:
        """Fetch and process web results with caching."""
        cache_key = f"search_{query}_{num_results}"
        cached = self.cache.get(cache_key)
        
        if cached:
            return [WebChunk(**chunk) for chunk in cached['chunks']]
        
        urls = list(search(query, num=num_results, stop=num_results, pause=2.0))
        chunks = []
        
        with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
            future_to_url = {executor.submit(fetch_url, url): url for url in urls}
            for future in as_completed(future_to_url):
                result = future.result()
                if not result:
                    continue
                    
                url, html = result
                try:
                    title, content = self.cleaner.extract_article_content(url, html)
                    text_chunks = chunk_text(content)
                    
                    for i, chunk_text in enumerate(text_chunks):
                        chunk = WebChunk(
                            text=chunk_text,
                            url=url,
                            title=title,
                            timestamp=datetime.now(),
                            chunk_index=i
                        )
                        chunks.append(chunk)
                except Exception as e:
                    logger.error(f"Error processing {url}: {e}")
        
        # Cache the results
        self.cache.set(cache_key, {
            'chunks': [vars(chunk) for chunk in chunks],
            'timestamp': datetime.now().isoformat()
        })
        
        return chunks

class RetrievalSystem:
    """Manages document indexing and retrieval."""
    def __init__(self, device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        self.cache = WebCache()
    
    def build_index(self, chunks: List[WebChunk]) -> Tuple[faiss.IndexFlatL2, np.ndarray]:
        """Build FAISS index from chunks with caching."""
        cache_key = f"index_{hash(''.join(c.text for c in chunks))}"
        cached = self.cache.get(cache_key)
        
        if cached:
            embeddings = np.array(cached['embeddings'])
            index = faiss.deserialize_index(bytes.fromhex(cached['index']))
            return index, embeddings
        
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        # Cache the results
        self.cache.set(cache_key, {
            'embeddings': embeddings.tolist(),
            'index': faiss.serialize_index(index).hex()
        })
        
        return index, embeddings
    
    def retrieve_context(self, query: str, chunks: List[WebChunk], 
                        index: faiss.IndexFlatL2, k: int = 3) -> List[Tuple[WebChunk, float]]:
        """Retrieve relevant chunks with similarity scores."""
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)
        distances, indices = index.search(query_embedding, k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            results.append((chunks[idx], float(dist)))
        
        return sorted(results, key=lambda x: x[1])

class ResponseGenerator:
    """Manages answer generation using the language model."""
    def __init__(self, model_name: str = Config.MODEL_NAME, token: str = Config.HF_TOKEN):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            use_auth_token=token,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        self.model.eval()
    
    def generate_answer(self, query: str, retrieved_chunks: List[Tuple[WebChunk, float]], 
                       max_new_tokens: int = 150) -> str:
        """Generate an answer with citations."""
        # Prepare context with citations
        context_parts = []
        citations = []
        
        for chunk, score in retrieved_chunks:
            context_parts.append(chunk.text)
            citations.append(chunk.get_citation())
        
        context = "\n\n".join(context_parts)
        
        # Construct prompt
        prompt = f"""Based on the following web sources:

{context}

Question: {query}

Please provide a detailed answer, citing specific sources when appropriate."""
        
        # Encode and potentially truncate input
        input_ids = self.tokenizer.encode(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=Config.MAX_TOKENS
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        answer = self.tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True)
        
        # Add citations
        return f"{answer.strip()}\n\nSources:\n" + "\n".join(f"- {citation}" for citation in citations)

def main():
    parser = argparse.ArgumentParser(description="Enhanced Web Retrieval Agent")
    parser.add_argument("--num_results", type=int, default=5, help="Number of web results to fetch")
    parser.add_argument("--top_k", type=int, default=3, help="Number of chunks to use for context")
    args = parser.parse_args()
    
    # Initialize components
    retriever = WebRetriever()
    retrieval_system = RetrievalSystem()
    generator = ResponseGenerator()
    
    print("\nEnhanced Web Agent Ready!")
    print("Enter your queries (type 'exit' to quit):")
    
    while True:
        try:
            query = input("\n>> ").strip()
            if query.lower() in ["exit", "quit"]:
                break
            
            # Fetch and process web content
            logger.info("Fetching web results...")
            chunks = retriever.get_web_results(query, num_results=args.num_results)
            if not chunks:
                print("No relevant content found.")
                continue
            
            # Build index and retrieve context
            logger.info("Building index and retrieving relevant chunks...")
            index, _ = retrieval_system.build_index(chunks)
            retrieved = retrieval_system.retrieve_context(query, chunks, index, k=args.top_k)
            
            # Generate answer
            logger.info("Generating response...")
            answer = generator.generate_answer(query, retrieved)
            
            print("\nResponse:")
            print("="*80)
            print(answer)
            print("="*80)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()