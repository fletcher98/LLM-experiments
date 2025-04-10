"""
Enhanced arXiv Agent with improved retrieval, caching, error handling, and extended functionality.
Features include document chunking, metadata extraction, citation generation, and query expansion.
The agent uses FAISS for efficient similarity search and Gemma for response generation.
"""

import argparse
import logging
import os
import re
import ast
import operator
import json
import time
from datetime import datetime, timedelta
from typing import List, Tuple, Any, Dict, Optional
from pathlib import Path
from functools import lru_cache
import hashlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import arxiv
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")
    MODEL_NAME: str = os.getenv("GEMMA_MODEL", "google/gemma-2-9b-it")
    CACHE_DIR: str = os.getenv("CACHE_DIR", "cache")
    MAX_CHUNK_SIZE: int = 512
    CACHE_EXPIRY_DAYS: int = 7
    MAX_RETRIES: int = 3
    MIN_RETRY_WAIT: float = 1.0
    MAX_RETRY_WAIT: float = 10.0

# Device selection with detailed logging
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    logger.info("Using MPS (Metal Performance Shaders) device")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device("cpu")
    logger.info("Using CPU device")

# Cache management
def setup_cache():
    """Initialize cache directory and clean old cache files."""
    cache_dir = Path(Config.CACHE_DIR)
    cache_dir.mkdir(exist_ok=True)
    
    # Clean old cache files
    expiry = datetime.now() - timedelta(days=Config.CACHE_EXPIRY_DAYS)
    for cache_file in cache_dir.glob("*.cache"):
        if datetime.fromtimestamp(cache_file.stat().st_mtime) < expiry:
            cache_file.unlink()
    
    logger.info(f"Cache directory set up at {cache_dir}")

class DocumentChunk:
    """Represents a chunk of text from an arXiv paper with metadata."""
    def __init__(self, text: str, paper_id: str, title: str, authors: List[str], 
                 publish_date: datetime, chunk_index: int):
        self.text = text
        self.paper_id = paper_id
        self.title = title
        self.authors = authors
        self.publish_date = publish_date
        self.chunk_index = chunk_index
        
    def get_citation(self) -> str:
        """Generate a citation string for the paper."""
        authors_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors_str += " et al."
        return f"{authors_str} ({self.publish_date.year}). {self.title}. arXiv:{self.paper_id}"

def chunk_document(text: str, chunk_size: int = Config.MAX_CHUNK_SIZE, 
                  overlap: int = 50) -> List[str]:
    """Split document into overlapping chunks for better retrieval."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

@retry(stop=stop_after_attempt(Config.MAX_RETRIES),
       wait=wait_exponential(multiplier=Config.MIN_RETRY_WAIT, max=Config.MAX_RETRY_WAIT))
def fetch_arxiv_papers(query: str, num_documents: int = 10, 
                      date_from: Optional[datetime] = None,
                      categories: Optional[List[str]] = None) -> List[DocumentChunk]:
    """
    Enhanced arXiv paper fetching with retry logic, filtering, and chunking.
    """
    try:
        # Construct advanced query
        full_query = query
        if categories:
            cat_query = " OR ".join(f"cat:{cat}" for cat in categories)
            full_query = f"({full_query}) AND ({cat_query})"
        if date_from:
            full_query = f"{full_query} AND submittedDate:[{date_from.strftime('%Y%m%d')}0000 TO *]"
        
        search = arxiv.Search(
            query=full_query,
            max_results=num_documents,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        
        chunks = []
        for result in search.results():
            # Extract metadata
            paper_id = result.get_short_id()
            authors = [author.name for author in result.authors]
            publish_date = result.published
            
            # Create document text and chunk it
            doc_text = f"Title: {result.title}\nAbstract: {result.summary}"
            doc_chunks = chunk_document(doc_text)
            
            # Create DocumentChunk objects
            for i, chunk_text in enumerate(doc_chunks):
                chunk = DocumentChunk(
                    text=chunk_text,
                    paper_id=paper_id,
                    title=result.title,
                    authors=authors,
                    publish_date=publish_date,
                    chunk_index=i
                )
                chunks.append(chunk)
        
        logger.info(f"Fetched and chunked {len(chunks)} segments from {num_documents} papers")
        return chunks
    
    except Exception as e:
        logger.error(f"Error fetching papers: {e}", exc_info=True)
        raise

def compute_cache_key(text: str) -> str:
    """Compute a cache key for a given text."""
    return hashlib.md5(text.encode()).hexdigest()

@lru_cache(maxsize=1000)
def get_embedding(text: str, embedder: SentenceTransformer) -> np.ndarray:
    """Cached computation of embeddings."""
    return embedder.encode(text, convert_to_numpy=True)

def build_index(chunks: List[DocumentChunk], embedder: SentenceTransformer) -> Tuple[faiss.IndexFlatL2, np.ndarray]:
    """Build FAISS index from document chunks with caching."""
    cache_file = Path(Config.CACHE_DIR) / f"index_{compute_cache_key(''.join(c.text for c in chunks))}.cache"
    
    if cache_file.exists():
        logger.info("Loading cached index")
        saved_data = np.load(cache_file, allow_pickle=True)
        embeddings = saved_data['embeddings']
        index = faiss.deserialize_index(saved_data['index'])
    else:
        # Generate embeddings with progress logging
        logger.info("Computing embeddings for chunks")
        embeddings = np.vstack([get_embedding(chunk.text, embedder) for chunk in chunks])
        
        # Build index
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        
        # Cache the results
        np.savez(cache_file, embeddings=embeddings, index=faiss.serialize_index(index))
        logger.info(f"Cached index to {cache_file}")
    
    return index, embeddings

def load_language_model(model_name: str = Config.MODEL_NAME, 
                       token: str = Config.HF_TOKEN) -> Tuple[AutoTokenizer, Any]:
    """Load and configure the language model with proper error handling."""
    try:
        logger.info(f"Loading language model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_auth_token=token,
            torch_dtype=torch.float16 if DEVICE.type != "cpu" else torch.float32,
            device_map="auto"
        ).to(DEVICE)
        model.eval()
        return tokenizer, model
    
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        raise

# Calculator operations (unchanged but with improved error messages)
OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}

def _safe_eval(node: ast.AST) -> float:
    """Safe evaluation of arithmetic expressions."""
    if isinstance(node, (ast.Num, ast.Constant)):
        return node.n if isinstance(node, ast.Num) else node.value
    elif isinstance(node, ast.BinOp):
        op_func = OPS.get(type(node.op))
        if op_func is None:
            raise TypeError(f"Unsupported operator: {type(node.op).__name__}")
        return op_func(_safe_eval(node.left), _safe_eval(node.right))
    elif isinstance(node, ast.UnaryOp):
        op_func = OPS.get(type(node.op))
        if op_func is None:
            raise TypeError(f"Unsupported operator: {type(node.op).__name__}")
        return op_func(_safe_eval(node.operand))
    raise TypeError(f"Unsupported expression type: {type(node).__name__}")

def simple_calculator(expression: str) -> Any:
    """Enhanced calculator with better error handling."""
    if not re.fullmatch(r'[\d\s\+\-\*\/\.\(\)]+', expression):
        return "Invalid expression: only numbers and basic arithmetic operators are allowed"
    try:
        result = _safe_eval(ast.parse(expression, mode="eval").body)
        return f"{result:.6g}"  # Format number for readability
    except Exception as e:
        return f"Error: {str(e)}"

def expand_query(query: str) -> List[str]:
    """Expand the query with related terms for better search."""
    expansions = [query]
    # Add variations
    words = query.lower().split()
    if len(words) > 1:
        # Add combinations of adjacent words
        for i in range(len(words) - 1):
            expansions.append(f"{words[i]} {words[i + 1]}")
    return expansions

def retrieve_context(query: str, index: faiss.IndexFlatL2, chunks: List[DocumentChunk],
                    embedder: SentenceTransformer, k: int = 3) -> List[Tuple[DocumentChunk, float]]:
    """Enhanced context retrieval with query expansion and relevance scores."""
    expanded_queries = expand_query(query)
    query_embedding = np.mean([get_embedding(q, embedder) for q in expanded_queries], axis=0)
    
    # Search index
    distances, indices = index.search(query_embedding.reshape(1, -1), k)
    
    # Return chunks with their distances (lower distance = more relevant)
    results = [(chunks[idx], dist) for idx, dist in zip(indices[0], distances[0])]
    return sorted(results, key=lambda x: x[1])  # Sort by relevance

def generate_answer(query: str, retrieved_chunks: List[Tuple[DocumentChunk, float]],
                   tokenizer: AutoTokenizer, model: Any,
                   max_new_tokens: int = 150) -> str:
    """Enhanced answer generation with citation support and better prompting."""
    # Prepare context with citations
    context_parts = []
    citations = []
    for chunk, score in retrieved_chunks:
        context_parts.append(chunk.text)
        citations.append(chunk.get_citation())
    
    context_text = "\n".join(context_parts)
    citations_text = "\nReferences:\n" + "\n".join(citations)
    
    # Construct prompt with specific instructions
    prompt = f"""Context: {context_text}

Question: {query}

Please provide a comprehensive answer based on the context above. Include relevant citations when referencing specific papers.

Answer:"""
    
    # Generate response
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    attention_mask = torch.ones_like(input_ids)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id
        )
    
    answer = tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True)
    return f"{answer.strip()}\n\n{citations_text}"

class ArXivAgent:
    """Enhanced arXiv agent with improved state management and caching."""
    
    def __init__(self, model_name: str = Config.MODEL_NAME):
        setup_cache()
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
        self.tokenizer, self.model = load_language_model(model_name)
        self.chunks: List[DocumentChunk] = []
        self.index: Optional[faiss.IndexFlatL2] = None
        
    def update_knowledge(self, query: str, num_documents: int = 10,
                        date_from: Optional[datetime] = None,
                        categories: Optional[List[str]] = None):
        """Update the agent's knowledge base with new papers."""
        self.chunks = fetch_arxiv_papers(query, num_documents, date_from, categories)
        self.index, _ = build_index(self.chunks, self.embedder)
    
    def answer_query(self, query: str, k: int = 3) -> str:
        """Process a query and generate an answer."""
        if query.lower().startswith("calc:"):
            expression = query[5:].strip()
            return f"Calculator result: {simple_calculator(expression)}"
        
        if not self.chunks or not self.index:
            raise ValueError("No documents loaded. Call update_knowledge first.")
        
        retrieved = retrieve_context(query, self.index, self.chunks, self.embedder, k)
        return generate_answer(query, retrieved, self.tokenizer, self.model)

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced arXiv Agent with advanced features and caching."
    )
    parser.add_argument("--query", type=str, default="machine learning",
                       help="Initial arXiv query to fetch papers")
    parser.add_argument("--num_documents", type=int, default=10,
                       help="Number of papers to fetch")
    parser.add_argument("--retrieval_k", type=int, default=3,
                       help="Number of chunks to retrieve for context")
    parser.add_argument("--days_back", type=int, default=None,
                       help="Only fetch papers from the last N days")
    parser.add_argument("--categories", type=str, nargs="+",
                       help="ArXiv categories to filter by (e.g., cs.AI cs.LG)")
    parser.add_argument("--model_name", type=str, default=Config.MODEL_NAME,
                       help="Name of the language model to use")
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = ArXivAgent(model_name=args.model_name)
    
    # Calculate date filter if specified
    date_from = None
    if args.days_back:
        date_from = datetime.now() - timedelta(days=args.days_back)
    
    # Update knowledge base
    logger.info("Initializing knowledge base...")
    agent.update_knowledge(
        args.query,
        num_documents=args.num_documents,
        date_from=date_from,
        categories=args.categories
    )
    
    # Interactive loop
    print("\nEnhanced arXiv Agent Ready!")
    print("Enter your queries (type 'exit' to quit):")
    while True:
        try:
            user_input = input(">> ").strip()
            if user_input.lower() in ["exit", "quit"]:
                break
            
            response = agent.answer_query(user_input, k=args.retrieval_k)
            print("\nAgent Answer:")
            print(response)
            print("\n" + "="*80 + "\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()