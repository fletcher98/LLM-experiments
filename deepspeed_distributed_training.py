"""
Enhanced distributed training demonstration using DeepSpeed for multi-node, 
multi-GPU neural network learning. Features include validation, checkpointing,
metrics tracking, and improved logging capabilities.
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import deepspeed
import time
from torch.utils.data import DataLoader, Dataset, random_split
from torch.distributed import barrier
import logging
import json
from pathlib import Path

# -----------------------------
# Setup Logging
# -----------------------------
def setup_logging(local_rank):
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - [%(local_rank)s] - %(message)s',
        level=logging.INFO if local_rank in [-1, 0] else logging.WARNING,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

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
        torch.manual_seed(42)  # For reproducibility
        self.data = torch.randn(num_samples, input_size)
        self.labels = torch.randint(0, num_classes, (num_samples,))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        return self.data[index], self.labels[index]

# -----------------------------
# Enhanced MLP Model
# -----------------------------
class EnhancedMLP(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[512, 256], num_classes=10, dropout_rate=0.1):
        super(EnhancedMLP, self).__init__()
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

# -----------------------------
# Metrics Calculator
# -----------------------------
class MetricsCalculator:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.running_loss = 0.0
        self.correct = 0
        self.total = 0
    
    def update(self, loss, outputs, labels):
        self.running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        self.total += labels.size(0)
        self.correct += (predicted == labels).sum().item()
    
    def get_metrics(self, step_count):
        avg_loss = self.running_loss / step_count
        accuracy = 100 * self.correct / self.total
        return {"loss": avg_loss, "accuracy": accuracy}

# -----------------------------
# Main Training Function
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepspeed_config", type=str, default="ds_config.json")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=5)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # Initialize distributed training
    deepspeed.init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    logger = setup_logging(local_rank)

    # Create checkpoint directory
    if local_rank <= 0:
        Path(args.checkpoint_dir).mkdir(exist_ok=True)

    # Load DeepSpeed config
    with open(args.deepspeed_config) as f:
        ds_config = json.load(f)

    # Create dataset and split into train/val
    dataset = DummyDataset(num_samples=10000, input_size=784, num_classes=10)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=ds_config["train_batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=ds_config["train_batch_size"], shuffle=False)

    # Initialize model and metrics
    model = EnhancedMLP(input_size=784, hidden_sizes=[512, 256], num_classes=10)
    train_metrics = MetricsCalculator()
    val_metrics = MetricsCalculator()

    # Initialize DeepSpeed engine
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters()
    )

    # Training loop
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        train_metrics.reset()
        model_engine.train()
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(model_engine.device)
            labels = labels.to(model_engine.device)

            # Forward pass
            outputs = model_engine(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)

            # Backward pass
            model_engine.backward(loss)
            model_engine.step()
            
            # Update metrics
            train_metrics.update(loss, outputs, labels)

            # Log progress
            if batch_idx % 50 == 0 and local_rank <= 0:
                logger.info(f"Train Epoch: {epoch+1} [{batch_idx}/{len(train_loader)}] "
                          f"Loss: {loss.item():.4f}")

        # Validation phase
        model_engine.eval()
        val_metrics.reset()
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(model_engine.device)
                labels = labels.to(model_engine.device)
                outputs = model_engine(inputs)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                val_metrics.update(loss, outputs, labels)

        # Get metrics
        train_results = train_metrics.get_metrics(len(train_loader))
        val_results = val_metrics.get_metrics(len(val_loader))

        # Log epoch results
        if local_rank <= 0:
            logger.info(
                f"Epoch {epoch+1}/{args.epochs} - "
                f"Train Loss: {train_results['loss']:.4f}, "
                f"Train Acc: {train_results['accuracy']:.2f}%, "
                f"Val Loss: {val_results['loss']:.4f}, "
                f"Val Acc: {val_results['accuracy']:.2f}%"
            )

        # Save checkpoint if validation accuracy improved
        if local_rank <= 0 and val_results["accuracy"] > best_val_acc:
            best_val_acc = val_results["accuracy"]
            checkpoint_path = os.path.join(args.checkpoint_dir, f"best_model.pt")
            client_state = {
                "epoch": epoch,
                "best_val_acc": best_val_acc,
                "train_metrics": train_results,
                "val_metrics": val_results
            }
            model_engine.save_checkpoint(args.checkpoint_dir, f"best_model", client_state=client_state)
            logger.info(f"Saved best model checkpoint with validation accuracy: {best_val_acc:.2f}%")

        # Regular checkpoint saving
        if (epoch + 1) % args.save_interval == 0 and local_rank <= 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
            model_engine.save_checkpoint(args.checkpoint_dir, f"checkpoint_epoch_{epoch+1}")
            logger.info(f"Saved regular checkpoint at epoch {epoch+1}")

        # Synchronize processes
        barrier()

if __name__ == "__main__":
    main()