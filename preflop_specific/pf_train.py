import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pf_poker_model import PokerNet
from shared.process_features import process_features
import glob

def pad_sequence(sequence, max_len=24):
    """Pad sequence to max_len with zeros"""
    curr_len = len(sequence)
    if curr_len >= max_len:
        return sequence[:max_len]  # Truncate if longer than max_len
    else:
        # Pad with zeros
        padding = torch.zeros((max_len - curr_len, sequence.shape[1]))
        return torch.cat([sequence, padding], dim=0)

def collate_poker_batch(batch):
    """Custom collate function to handle variable length sequences"""
    # Find max sequence length in this batch
    max_len = max(b['action_sequence'].shape[0] for b in batch)
    max_len = min(max_len, 24)  # Cap at 24 actions to prevent excessive padding
    
    # Get original lengths and pad sequences
    lengths = torch.tensor([len(b['action_sequence']) for b in batch])
    
    # Sort by sequence length in descending order (required for pack_padded_sequence)
    lengths, sort_idx = lengths.sort(descending=True)
    
    # Sort and pad all sequences
    padded_sequences = []
    sorted_static = []
    sorted_labels = []
    sorted_sizes = []
    
    for idx in sort_idx:
        b = batch[idx]
        # Pad sequence
        padded = pad_sequence(b['action_sequence'], max_len)
        padded_sequences.append(padded)
        # Sort other elements to match
        sorted_static.append(b['static_features'])
        sorted_labels.append(b['action_label'])
        sorted_sizes.append(b['size_label'])
    
    # Stack all tensors
    return {
        'static_features': torch.stack(sorted_static),
        'action_sequence': torch.stack(padded_sequences),
        'sequence_lengths': lengths,
        'action_label': torch.stack(sorted_labels),
        'size_label': torch.stack(sorted_sizes)
    }

class PokerDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Process the row into features
        static_features, action_sequence = process_features(self.data[idx], 0)
        
        # Get the target (what action was actually taken)
        actual_action = self.data[idx]['label']
        target_action_map = {
            'folds': 0,
            'calls': 1,
            'raises': 2,
            'calls all-in': 3,
            'raises all-in': 4,
            'checks': 5
        }  # Maps to output neurons in model's action_head
        action_label = target_action_map[actual_action['action']]
        
        # Get the size label (if applicable)
        size_label = actual_action['size'] or 0
        size_label = float(size_label) / self.data[idx]['preflop_gamestate']['blinds']  # Normalize by blinds

        return {
            'static_features': static_features,
            'action_sequence': action_sequence,
            'action_label': torch.tensor(action_label),
            'size_label': torch.tensor(size_label).float()
        }

def train_model(model, train_loader, val_loader=None, epochs=20, lr=0.0005, device='cuda'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    action_criterion = nn.CrossEntropyLoss()
    size_criterion = nn.HuberLoss()  # Changed to Huber loss for better raise size prediction
    
    model = model.to(device)
    best_val_accuracy = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            # Move data to device
            static_features = batch['static_features'].to(device)
            action_sequence = batch['action_sequence'].to(device)
            sequence_lengths = batch['sequence_lengths'].to(device)
            action_labels = batch['action_label'].to(device)
            size_labels = batch['size_label'].to(device)
            
            # Forward pass with sequence lengths
            action_logits, size_pred = model(
                static_features, 
                action_sequence,
                sequence_lengths
            )
            
            # Calculate action loss
            action_loss = action_criterion(action_logits, action_labels)
            
            # Calculate size loss only for raise actions
            raise_mask = (action_labels == 2) | (action_labels == 4)  # Include both raise and raise-all-in
            size_loss = 0
            if raise_mask.any():
                # Get actual raise sizes and predictions for raise actions
                raise_sizes = size_labels[raise_mask]
                raise_preds = size_pred[raise_mask]
                
                # Ensure both tensors have the same shape
                if len(raise_preds.shape) > 1:
                    raise_preds = raise_preds.squeeze(-1)
                
                # Calculate size loss only for valid raise sizes (> 0)
                valid_raise_mask = raise_sizes > 0
                if valid_raise_mask.any():
                    size_loss = size_criterion(
                        raise_preds[valid_raise_mask],
                        raise_sizes[valid_raise_mask]
                    )
            
            # Combined loss with balanced weights
            loss = action_loss + size_loss  # Equal weight for both losses
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f'Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}')
        
        # Validation
        if val_loader:
            model.eval()
            correct = 0
            total = 0
            val_loss = 0
            raise_mae = 0.0  # Track Mean Absolute Error for raise sizes
            num_raises = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    static_features = batch['static_features'].to(device)
                    action_sequence = batch['action_sequence'].to(device)
                    sequence_lengths = batch['sequence_lengths'].to(device)
                    action_labels = batch['action_label'].to(device)
                    size_labels = batch['size_label'].to(device)
                    
                    action_logits, size_pred = model(static_features, action_sequence, sequence_lengths)
                    _, predicted = torch.max(action_logits, 1)
                    
                    # Calculate validation metrics
                    action_loss = action_criterion(action_logits, action_labels)
                    
                    # Calculate raise size accuracy
                    raise_mask = (action_labels == 2) | (action_labels == 4)
                    if raise_mask.any():
                        raise_sizes = size_labels[raise_mask]
                        raise_preds = size_pred[raise_mask]
                        
                        # Ensure both tensors have the same shape
                        if len(raise_preds.shape) > 1:
                            raise_preds = raise_preds.squeeze(-1)
                        
                        valid_raise_mask = raise_sizes > 0
                        if valid_raise_mask.any():
                            mae = torch.abs(raise_preds[valid_raise_mask] - raise_sizes[valid_raise_mask]).mean()
                            raise_mae += mae.item()
                            num_raises += 1
                    
                    val_loss += action_loss.item()
                    total += action_labels.size(0)
                    correct += (predicted == action_labels).sum().item()
            
            val_accuracy = 100 * correct / total
            avg_val_loss = val_loss / len(val_loader)
            avg_raise_mae = raise_mae / num_raises if num_raises > 0 else 0
            
            print(f'Validation Accuracy: {val_accuracy:.2f}%, Validation Loss: {avg_val_loss:.4f}')
            print(f'Average Raise Size MAE: {avg_raise_mae:.2f} BB')
            
            # Update learning rate based on validation loss
            scheduler.step(avg_val_loss)
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model_state = model.state_dict().copy()
    
    # Restore best model if we have one
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f'Restored best model with validation accuracy: {best_val_accuracy:.2f}%')

def main():
    # Load all parquet files from directory
    parquet_files = glob.glob("../data/preflop/*.parquet")
    print(f"Found {len(parquet_files)} parquet files")
    
    # Load and combine all dataframes
    dfs = []
    for file in parquet_files:
        print(f"Loading {file}...")
        df = pd.read_parquet(file)
        dfs.append(df)
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Total number of samples: {len(combined_df)}")
    
    # Convert to list of records
    data_list = combined_df.to_dict(orient="records")
    
    # Create dataset
    dataset = PokerDataset(data_list)
    
    # Split into train/val
    train_size = int(0.91 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    
    # Create dataloaders with custom collate function
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True,
        collate_fn=collate_poker_batch
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32, 
        shuffle=False,
        collate_fn=collate_poker_batch
    )
    
    # Initialize model with correct dimensions
    model = PokerNet(
        static_dim=19,  # From process_features
        action_dim=4,   # [action_type, player, amount]
        hidden_dim=256,  # Increased from 128
        gru_hidden_dim=128  # Increased from 64
    )
    
    # Train
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_model(model, train_loader, val_loader, epochs=4, device=device)
    
    # Save model
    torch.save(model.state_dict(), '../models/poker_model_pf.pth')
    print("Model saved to poker_model_pf.pth")

if __name__ == "__main__":
    main() 