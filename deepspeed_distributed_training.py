"""
This project is a distributed training demonstration using DeepSpeed for multi-node, 
multi-GPU neural network learning. It integrates a simple MLP model with a dummy dataset 
and leverages DeepSpeed’s advanced optimization techniques such as ZeRO and FP16 to 
showcase scalable and efficient deep learning training.
""" 

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import deepspeed
from torch.utils.data import DataLoader, Dataset

# -----------------------------
# Dummy Dataset
# -----------------------------
class DummyDataset(Dataset):
    def __init__(self, num_samples=1000, input_size=784, num_classes=10):
        super(DummyDataset, self).__init__()
        self.num_samples = num_samples
        self.input_size = input_size
        self.num_classes = num_classes
        # Generate random data and labels
        self.data = torch.randn(num_samples, input_size)
        self.labels = torch.randint(0, num_classes, (num_samples,))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        return self.data[index], self.labels[index]

# -----------------------------
# Simple MLP Model
# -----------------------------
class SimpleMLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)

# -----------------------------
# Main Training Function
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    # Path to the DeepSpeed configuration file
    parser.add_argument("--deepspeed_config", type=str, default="ds_config.json", help="DeepSpeed config file path")
    # Additional DeepSpeed config arguments can be added by DeepSpeed helper
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # Initialize distributed training (multi-node/multi-GPU)
    deepspeed.init_distributed()

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create dataset and dataloader
    dataset = DummyDataset(num_samples=1000, input_size=784, num_classes=10)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize the model
    model = SimpleMLP(input_size=784, hidden_size=256, num_classes=10)

    # Setup optimizer (Adam)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Initialize DeepSpeed engine
    model, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer,
        model_parameters=model.parameters()
    )

    # Training loop
    model.train()
    num_epochs = 5
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            inputs, labels = batch
            # Move data to the same device as the model (managed by DeepSpeed)
            inputs = inputs.to(model.device)
            labels = labels.to(model.device)

            # Forward pass
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)

            # Backward pass with DeepSpeed engine (handles gradient scaling, etc.)
            model.backward(loss)
            model.step()
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs} completed. Average Loss: {epoch_loss/len(dataloader):.4f}")

if __name__ == "__main__":
    main()