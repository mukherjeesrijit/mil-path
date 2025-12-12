"""
solver.py
Training and validation logic for MIL models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from pathlib import Path


class MILSolver:
    """
    Solver for training and validating MIL models.
    """
    
    def __init__(self, model, device='cuda', learning_rate=1e-3, 
                 save_dir='checkpoints'):
        """
        Args:
            model: MIL model instance
            device: 'cuda' or 'cpu'
            learning_rate: Learning rate for optimizer
            save_dir: Directory to save model checkpoints
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.learning_rate = learning_rate
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Loss and optimizer
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': []
        }
        
        print(f"MILSolver initialized on device: {self.device}")
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            avg_loss: Average training loss for the epoch
        """
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(train_loader, desc="Training", leave=False):
            patches = batch['patches'].to(self.device)
            label = batch['label'].float().unsqueeze(1).to(self.device)
            
            # Forward pass
            patches = patches.squeeze(0)  # Remove batch dimension
            output = self.model(patches)
            
            # Handle output shape
            if output.dim() == 1:
                output = output.unsqueeze(0)
            
            loss = self.criterion(output, label)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        return avg_loss
    
    def validate(self, val_loader):
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            tuple: (avg_val_loss, accuracy)
        """
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                patches = batch['patches'].to(self.device)
                label = batch['label'].float().unsqueeze(1).to(self.device)
                
                # Forward pass
                patches = patches.squeeze(0)
                output = self.model(patches)
                
                # Handle output shape
                if output.dim() == 1:
                    output = output.unsqueeze(0)
                
                loss = self.criterion(output, label)
                val_loss += loss.item()
                
                # Compute predictions
                preds = torch.sigmoid(output) > 0.5
                correct += (preds.float() == label).sum().item()
                total += label.size(0)
        
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        accuracy = correct / total if total > 0 else 0
        
        return avg_val_loss, accuracy
    
    def train(self, train_loader, val_loader, num_epochs=10, 
              save_best=True, verbose=True):
        """
        Full training loop with validation.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            save_best: Whether to save the best model
            verbose: Whether to print training progress
            
        Returns:
            history: Dictionary with training history
        """
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print progress
            if verbose:
                print(f"Epoch {epoch+1}/{num_epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val Acc: {val_acc:.4f}")
            
            # Save best model
            if save_best and val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, 'best_model.pth')
                if verbose:
                    print(f"  → Best model saved (val_loss: {val_loss:.4f})")
        
        return self.history
    
    def save_checkpoint(self, epoch, filename='checkpoint.pth'):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            filename: Checkpoint filename
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }
        filepath = self.save_dir / filename
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filename='checkpoint.pth'):
        """
        Load model checkpoint.
        
        Args:
            filename: Checkpoint filename
        """
        filepath = self.save_dir / filename
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        
        print(f"Checkpoint loaded from {filepath}")
        return checkpoint['epoch']
    
    def predict(self, patches):
        """
        Make prediction on a bag of patches.
        
        Args:
            patches: Tensor of patches [K, C, H, W]
            
        Returns:
            tuple: (probability, prediction)
        """
        self.model.eval()
        with torch.no_grad():
            patches = patches.to(self.device)
            output = self.model(patches)
            prob = torch.sigmoid(output).item()
            pred = int(prob > 0.5)
        return prob, pred


def train_mil_model(model, train_loader, val_loader, 
                   num_epochs=10, learning_rate=1e-3, 
                   device='cuda', save_dir='checkpoints'):
    """
    Convenience function to train a MIL model.
    
    Args:
        model: MIL model instance
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs
        learning_rate: Learning rate
        device: Device to use
        save_dir: Directory to save checkpoints
        
    Returns:
        tuple: (trained_model, history)
    """
    solver = MILSolver(
        model=model,
        device=device,
        learning_rate=learning_rate,
        save_dir=save_dir
    )
    
    history = solver.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        save_best=True,
        verbose=True
    )
    
    return solver.model, history


if __name__ == "__main__":
    print("MIL Solver Module")
    print("=" * 60)
    print("\nUsage example:")
    print("""
    from model import create_mil_model
    from dataset import create_mil_dataloaders
    from solver import train_mil_model
    
    # Create model
    model = create_mil_model(model_type='attention', num_classes=1)
    
    # Create dataloaders
    train_loader, val_loader = create_mil_dataloaders(
        train_csv='wsi_train_labels.csv',
        val_csv='wsi_val_labels.csv'
    )
    
    # Train model
    trained_model, history = train_mil_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=10,
        learning_rate=1e-3
    )
    """)