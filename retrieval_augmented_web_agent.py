"""
This project focuses on retrieving web documents using Google search based on a given query.
It fetches webpages, extracts their textual content using BeautifulSoup, and aggregates them into documents.
The documents are then encoded using a SentenceTransformer and indexed with FAISS for efficient retrieval.
For a new query, the system retrieves the most relevant documents to provide context, constructs a prompt,
and generates an answer using the Gemma language model via Hugging Face's pipeline.
"""

import torch
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')
import os
import requests
from bs4 import BeautifulSoup
from googlesearch import search
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Hugging Face authentication token and model
hf_token = "YOUR_HF_TOKEN"
model_name = "YOUR_MODEL_NAME"

def get_web_results(query, num_results=5):
    """
    Fetch URLs using googlesearch and extract webpage text with BeautifulSoup.
    """
    # Use 'num' and 'stop' to control the number of results.
    urls = list(search(query, num=num_results, stop=num_results, pause=2.0))
    results = []
    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                text = soup.get_text(separator=' ', strip=True)
                results.append(text)
        except Exception as e:
            print(f"Error fetching {url}: {e}")
    return results

def build_faiss_index(documents, embedder):
    """
    Compute embeddings for documents and build a FAISS index.
    """
    # Use num_workers=0 to avoid multiprocessing issues.
    embeddings = embedder.encode(documents, convert_to_numpy=True, num_workers=0)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings

def retrieve_context(query, documents, index, embedder, k=2):
    """
    Retrieve the top k documents most relevant to the query.
    """
    query_embedding = embedder.encode([query], convert_to_numpy=True, num_workers=0)
    distances, indices = index.search(query_embedding, k)
    retrieved_docs = [documents[i] for i in indices[0]]
    return retrieved_docs

def main():
    query = input("Enter your query: ")
    print("Fetching web results...")
    documents = get_web_results(query, num_results=5)
    if not documents:
        print("No documents retrieved.")
        return
    print(f"Retrieved {len(documents)} documents.")

    # Initialize the SentenceTransformer embedding model.
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    index, _ = build_faiss_index(documents, embedder)
    retrieved_docs = retrieve_context(query, documents, index, embedder, k=2)
    
    print("\nTop retrieved context (first 200 characters each):")
    for i, doc in enumerate(retrieved_docs, start=1):
        print(f"Document {i}: {doc[:200]}...\n")
    
    # Construct the prompt from the retrieved documents.
    context_text = "\n".join(retrieved_docs)
    prompt = f"Context: {context_text}\nQuery: {query}\nAnswer:"

    # Load the Gemma model and tokenizer using your Hugging Face token.
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token).to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a text-generation pipeline.
    gen_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
    
    # Truncate the prompt to fit within Gemma's maximum input length (typically 1024 tokens).
    encoded_prompt = tokenizer.encode(prompt, truncation=True, max_length=1024)
    prompt = tokenizer.decode(encoded_prompt, skip_special_tokens=True)
    
    # Generate new tokens; using max_new_tokens to control the generation length.
    result = gen_pipeline(prompt, max_new_tokens=50, truncation=True, num_return_sequences=1)
    answer = result[0]['generated_text']
    
    # Remove the prompt portion from the output if it's repeated.
    if answer.startswith(prompt):
        answer = answer[len(prompt):].strip()
    
    print("\nGenerated Answer:")
    print(answer)

if __name__ == "__main__":
    main()