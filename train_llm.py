"""
This end-to-end project demonstrates the process of training a MiniGPT model 
on the WikiText-103 dataset. It begins with data preparation, where the 
dataset is tokenized using GPT-2’s tokenizer and segmented into fixed-length 
sequences using a custom PyTorch Dataset and DataLoader. The MiniGPT model, 
built with transformer blocks, is trained using an Adam optimizer and 
cross-entropy loss, with training progress monitored through loss and 
perplexity metrics on both training and validation data. After the training
loop completes, the model and tokenizer are saved, providing a full pipeline 
from raw text data to a deployable language model.
"""

import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import GPT2Tokenizer
from tqdm import tqdm  # for progress bars

# -------------------------------
# Hyperparameters
# -------------------------------
batch_size = 16
epochs = 10                
learning_rate = 3e-4
seq_length = 128           # training sequence length
d_model = 256              # increased embedding dimension
n_heads = 8                # increased number of attention heads
n_layers = 4               # increased number of transformer layers
d_ff = 512                 # increased feed-forward dimension
dropout = 0.1

# -------------------------------
# Dataset Preparation
# -------------------------------
class TextDataset(Dataset):
    def __init__(self, token_ids, seq_length):
        self.seq_length = seq_length
        # Create non-overlapping chunks
        self.data = [token_ids[i:i+seq_length] 
                     for i in range(0, len(token_ids) - seq_length, seq_length)]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long)

# -------------------------------
# Causal Mask for Self-Attention
# -------------------------------
def generate_causal_mask(seq_len):
    return torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)

# -------------------------------
# Transformer Block Definition
# -------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        seq_len = x.size(0)
        mask = generate_causal_mask(seq_len).to(x.device)
        attn_output, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.ln1(x + attn_output)
        ff_output = self.ff(x)
        x = self.ln2(x + ff_output)
        return x

# -------------------------------
# Minimal GPT-style Transformer Model
# -------------------------------
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_seq_length, dropout):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # Learnable positional embeddings
        self.pos_embedding = nn.Parameter(torch.zeros(max_seq_length, d_model))
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        # x shape: (batch, seq_length)
        batch, seq_length = x.size()
        token_emb = self.token_embedding(x)  # (batch, seq_length, d_model)
        pos_emb = self.pos_embedding[:seq_length].unsqueeze(0)  # (1, seq_length, d_model)
        x = token_emb + pos_emb
        # Transformer expects (seq_length, batch, d_model)
        x = x.transpose(0, 1)
        for layer in self.layers:
            x = layer(x)
        x = x.transpose(0, 1)  # back to (batch, seq_length, d_model)
        x = self.ln_f(x)
        logits = self.fc_out(x)
        return logits

# -------------------------------
# Main Training Loop
# -------------------------------
def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load WikiText-103 dataset and tokenizer
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = int(1e12)

    # Batch tokenization for training data
    print("Tokenizing training data...")
    texts = dataset["train"]["text"]
    batch_size = 100
    train_tokens = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Batch tokenizing train"):
        batch_texts = texts[i:i+batch_size]
        batch_tokens = tokenizer.batch_encode_plus(batch_texts, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        for tokens in batch_tokens:
            train_tokens.extend(tokens + [tokenizer.eos_token_id])
    
    print(f"Training tokenization complete. Total tokens: {len(train_tokens)}")

    # Batch tokenization for validation data
    print("Tokenizing validation data...")
    texts = dataset["validation"]["text"]
    val_tokens = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Batch tokenizing val"):
        batch_texts = texts[i:i+batch_size]
        batch_tokens = tokenizer.batch_encode_plus(batch_texts, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        for tokens in batch_tokens:
            val_tokens.extend(tokens + [tokenizer.eos_token_id])
    
    print(f"Validation tokenization complete. Total tokens: {len(val_tokens)}")

    # Create PyTorch datasets and dataloaders
    train_dataset = TextDataset(train_tokens, seq_length)
    val_dataset = TextDataset(val_tokens, seq_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    vocab_size = tokenizer.vocab_size
    model = MiniGPT(vocab_size, d_model, n_layers, n_heads, d_ff, seq_length, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training loop with progress bars
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        train_batches = tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False)
        for batch in train_batches:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)  # (batch, seq_length, vocab_size)
            logits = logits[:, :-1, :].contiguous()  # shift logits
            targets = batch[:, 1:].contiguous()        # shift targets
            loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_batches.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Training Loss: {avg_loss:.4f}")
        
        # Validation loop with progress bar
        model.eval()
        total_val_loss = 0.0
        val_batches = tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", leave=False)
        with torch.no_grad():
            for batch in val_batches:
                batch = batch.to(device)
                logits = model(batch)
                logits = logits[:, :-1, :].contiguous()
                targets = batch[:, 1:].contiguous()
                loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
                total_val_loss += loss.item()
                val_batches.set_postfix(loss=loss.item())
        
        avg_val_loss = total_val_loss / len(val_loader)
        perplexity = math.exp(avg_val_loss)
        print(f"Epoch {epoch+1} - Validation Loss: {avg_val_loss:.4f} - Perplexity: {perplexity:.4f}")

    # Save model checkpoint and tokenizer
    os.makedirs("checkpoint", exist_ok=True)
    torch.save(model.state_dict(), "checkpoint/minigpt.pt")
    tokenizer.save_pretrained("checkpoint")
    print("Model and tokenizer saved in the 'checkpoint/' directory.")

if __name__ == "__main__":
    main()