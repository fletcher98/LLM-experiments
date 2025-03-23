"""
This project focuses on fetching academic papers from arXiv by using
its API to collect papers based on a given query. It constructs documents 
by combining each paper’s title and abstract, then computes embeddings with 
a SentenceTransformer and indexes them using FAISS. For a new query, the system
retrieves the most relevant documents to provide context. Additionally, it
incorporates a simple calculator tool: if the query starts with a designated prefix 
(e.g., “calc:”), the system evaluates arithmetic expressions directly. Otherwise, it 
combines the retrieved academic context with the query and generates an answer using
the Gemma model. An evaluation routine further demonstrates the agent’s capabilities 
on a set of predefined queries.
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import arxiv
import re

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set your Hugging Face token and model name for Gemma
hf_token = "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"
default_model_name = "google/gemma-2-9b-it"

# --- Step 1: ArXiv Scraper ---
def fetch_arxiv_papers(query: str, num_documents: int = 10):
    """
    Fetches papers from arXiv based on the given query.
    Returns a list of documents combining the title and abstract.
    """
    try:
        search = arxiv.Search(
            query=query,
            max_results=num_documents,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        documents = []
        for result in search.results():
            doc = f"Title: {result.title}. Abstract: {result.summary}"
            documents.append(doc)
        print(f"Fetched {len(documents)} papers from arXiv.")
        return documents
    except Exception as e:
        print(f"Error fetching papers: {e}")
        return []

# --- Step 2: Build the FAISS Index over ArXiv Documents ---
def build_index(documents, embedder):
    """
    Given a list of documents and a SentenceTransformer embedder,
    build a FAISS index for retrieval.
    """
    # Generate document embeddings
    doc_embeddings = embedder.encode(documents, convert_to_numpy=True)
    dim = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(doc_embeddings)
    print(f"Indexed {len(documents)} documents from arXiv.")
    return index, doc_embeddings

# --- Step 3: Set up the language model (Gemma) ---
def load_language_model(model_name=default_model_name, token=hf_token):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=token).to(DEVICE)
    model.eval()
    return tokenizer, model

# --- Step 4: Define a simple tool integration (calculator) ---
def simple_calculator(expression):
    """
    A simple calculator that evaluates basic arithmetic expressions.
    E.g., "2 + 3 * 4" -> 14
    """
    try:
        if re.fullmatch(r'[\d\s\+\-\*\/\.\(\)]+', expression):
            result = eval(expression)
            return result
        else:
            return "Invalid expression."
    except Exception as e:
        return f"Error: {e}"

# --- Step 5: Retrieval augmented generation using ArXiv documents ---
def retrieve_context(query: str, index, documents, embedder, k: int = 2):
    """
    Given a query, retrieves k relevant documents from the FAISS index.
    """
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    retrieved = [documents[idx] for idx in indices[0]]
    return retrieved

def generate_answer(query: str, retrieved_context, tokenizer, model, max_new_tokens: int = 150):
    """
    Uses Gemma to generate an answer by combining retrieved context with the query.
    """
    context_text = " ".join(retrieved_context)
    prompt = f"Context: {context_text}\nQuestion: {query}\nAnswer:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            pad_token_id=tokenizer.pad_token_id
        )
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return answer

# --- Step 6: The Agent ---
def agent(query: str, index, documents, embedder, tokenizer, model, k: int):
    """
    Processes the query: if it starts with "calc:" then uses the calculator;
    otherwise, retrieves relevant arXiv documents and generates an answer.
    """
    if query.lower().startswith("calc:"):
        expression = query[5:].strip()
        result = simple_calculator(expression)
        return f"Calculator result: {result}"
    else:
        retrieved = retrieve_context(query, index, documents, embedder, k)
        answer = generate_answer(query, retrieved, tokenizer, model)
        return answer

# --- Step 7: Evaluation / Benchmarking Routine ---
def evaluate_agent(test_queries, index, documents, embedder, tokenizer, model, k: int):
    results = {}
    for q in test_queries:
        results[q] = agent(q, index, documents, embedder, tokenizer, model, k)
    return results

# --- Main Function with CLI Interface ---
def main():
    parser = argparse.ArgumentParser(
        description="CLI for an arXiv Scraper LLM Agent with External Tool Integration."
    )
    parser.add_argument(
        "--query",
        type=str,
        default="machine learning",
        help="The arXiv query to fetch papers (default: 'machine learning')."
    )
    parser.add_argument(
        "--num_documents",
        type=int,
        default=None,
        help="Number of arXiv papers to fetch. If not provided, you will be prompted."
    )
    parser.add_argument(
        "--retrieval_k",
        type=int,
        default=2,
        help="Number of documents to retrieve for context (default: 2)."
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation on a set of test queries."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=default_model_name,
        help="Name of the language model to use (default: google/gemma-2-9b-it)."
    )
    args = parser.parse_args()

    # Prompt for number of documents if not provided via CLI
    if args.num_documents is None:
        try:
            num_docs = int(input("Enter the number of papers to fetch: "))
        except ValueError:
            print("Invalid input. Using default of 10 documents.")
            num_docs = 10
    else:
        num_docs = args.num_documents

    # Fetch papers from arXiv using the provided number of documents
    print("Fetching arXiv papers...")
    documents = fetch_arxiv_papers(args.query, num_documents=num_docs)
    if not documents:
        print("No documents fetched. Check your query or network connection.")
        return

    # Initialize embedder and language model
    embedder = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
    tokenizer, model = load_language_model(args.model_name, token=hf_token)

    # Build the FAISS index
    index, _ = build_index(documents, embedder)

    if args.evaluate:
        test_queries = [
            "What recent advances in machine learning are mentioned in these papers?",
            "Explain a method for training deep learning models.",
            "calc: 45 / (3 + 2)",
            "Summarize the key findings from machine learning research."
        ]
        results = evaluate_agent(test_queries, index, documents, embedder, tokenizer, model, args.retrieval_k)
        for query, answer in results.items():
            print("\nQuery:", query)
            print("Agent Answer:", answer)
    else:
        print("\nEnter your queries (type 'exit' to quit):")
        while True:
            user_input = input(">> ")
            if user_input.lower() in ["exit", "quit"]:
                break
            response = agent(user_input, index, documents, embedder, tokenizer, model, args.retrieval_k)
            print("Agent Answer:", response)

if __name__ == "__main__":
    main()