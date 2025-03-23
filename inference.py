"""
Extension of train_llm.py. Built a lightweight GPT-style transformer model 
from scratch using PyTorch, which includes essential components such as token 
and positional embeddings, multiple transformer blocks (each with self-attention 
and feed-forward layers), and a final linear layer for output generation. A causal 
mask is implemented to ensure the autoregressive nature of the model during text 
generation. An interactive loop lets users input prompts and receive generated text, 
showcasing how a minimal transformer architecture can be used for creative text 
generation and experimentation.
"""

import torch
import torch.nn as nn
from transformers import GPT2Tokenizer

# -------------------------------
# Utility: Causal Mask
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
# Minimal GPT-style Model Definition
# -------------------------------
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_seq_length, dropout):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(max_seq_length, d_model))
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        batch, seq_length = x.size()
        token_emb = self.token_embedding(x)
        pos_emb = self.pos_embedding[:seq_length].unsqueeze(0)
        x = token_emb + pos_emb
        x = x.transpose(0, 1)  # (seq_length, batch, d_model)
        for layer in self.layers:
            x = layer(x)
        x = x.transpose(0, 1)  # (batch, seq_length, d_model)
        x = self.ln_f(x)
        logits = self.fc_out(x)
        return logits

# -------------------------------
# Text Generation Function
# -------------------------------
def generate_text(model, tokenizer, prompt, max_new_tokens=50, device="cpu"):
    model.eval()
    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if input_ids.size(1) > model.pos_embedding.size(0):
                input_ids = input_ids[:, -model.pos_embedding.size(0):]
            logits = model(input_ids)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

# -------------------------------
# Interactive Inference Loop
# -------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = GPT2Tokenizer.from_pretrained("checkpoint")
    vocab_size = tokenizer.vocab_size

    # Hyperparameters must match training settings
    d_model = 256
    n_layers = 4
    n_heads = 8
    d_ff = 512
    seq_length = 128
    dropout = 0.1

    model = MiniGPT(vocab_size, d_model, n_layers, n_heads, d_ff, seq_length, dropout).to(device)
    model.load_state_dict(torch.load("checkpoint/minigpt.pt", map_location=device))
    
    print("Interactive text generation (type 'exit' to quit).")
    while True:
        prompt = input("Prompt: ")
        if prompt.strip().lower() == "exit":
            break
        generated = generate_text(model, tokenizer, prompt, max_new_tokens=50, device=device)
        print("\nGenerated text:")
        print(generated)
        print("-" * 50)

if __name__ == "__main__":
    main()