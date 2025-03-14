import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

class PokerNet(nn.Module):
    def __init__(self, static_dim=19, action_dim=3, hidden_dim=256, gru_hidden_dim=128):
        """
        Parameters:
        - static_dim: dimension of static features (19 in our case from process_features)
        - action_dim: dimension of each action (3 in our case: action_type, player, amount)
        - hidden_dim: dimension of hidden layers
        - gru_hidden_dim: dimension of GRU hidden state
        """
        super().__init__()
        
        # Process static features with a deeper network
        self.static_net = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.2)
        )
        
        # Process action sequence with deeper GRU
        self.gru = nn.GRU(
            input_size=action_dim,  # (action_type, player, amount)
            hidden_size=gru_hidden_dim,
            num_layers=3,  # Increased number of layers
            batch_first=True,
            dropout=0.2,
            bidirectional=True  # Make it bidirectional for better sequence understanding
        )
        
        # Attention layer for GRU outputs
        self.attention = nn.Sequential(
            nn.Linear(gru_hidden_dim * 2, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Combine static and sequential features
        combined_dim = (hidden_dim // 2) + (gru_hidden_dim * 2)  # *2 for bidirectional
        
        # Deeper decision layers
        self.decision_layers = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.2)
        )
        
        # Output heads with additional layers
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, 6)  # 6 final-result actions types (no post)
        )
        
        # Modified size head to predict raise sizes better
        self.size_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 8),
            nn.Linear(hidden_dim // 8, 1),
            nn.Softplus()  # Changed from ReLU to Softplus for smoother positive outputs
        )
        
    def forward(self, static_features, action_sequence, sequence_lengths):
        """
        Parameters:
        - static_features: tensor of shape (batch_size, static_dim)
        - action_sequence: tensor of shape (batch_size, seq_len, action_dim)
        - sequence_lengths: tensor of shape (batch_size,) with actual lengths
        
        Returns:
        - action_logits: probabilities for actions
        - size_pred: predicted bet/raise size (in BB)
        """
        # Process static features
        static_out = self.static_net(static_features)
        
        # Ensure sequence lengths don't exceed sequence size
        max_seq_len = action_sequence.size(1)
        sequence_lengths = torch.clamp(sequence_lengths, max=max_seq_len)
        
        # Pack padded sequence for GRU
        try:
            packed_sequence = rnn_utils.pack_padded_sequence(
                action_sequence, 
                sequence_lengths.cpu(),
                batch_first=True,
                enforce_sorted=True
            )
            
            # Process action sequence
            packed_output, _ = self.gru(packed_sequence)
            
            # Unpack the sequence
            output, _ = rnn_utils.pad_packed_sequence(packed_output, batch_first=True)
            
            # Apply attention to get weighted sum of all hidden states
            attention_weights = self.attention(output)  # (batch_size, seq_len, 1)
            attention_weights = torch.softmax(attention_weights, dim=1)  # normalize over sequence length
            sequence_out = torch.bmm(attention_weights.transpose(1, 2), output).squeeze(1)  # (batch_size, hidden_size*2)
            
        except Exception as e:
            print(f"Error in processing sequence: {e}")
            print(f"Sequence shape: {action_sequence.shape}")
            print(f"Lengths: {sequence_lengths}")
            raise e
        
        # Combine features
        combined = torch.cat([static_out, sequence_out], dim=1)
        
        # Final processing
        features = self.decision_layers(combined)
        
        # Generate outputs
        action_logits = self.action_head(features)
        
        # Generate size prediction with minimum raise of 2BB
        raw_size = self.size_head(features)
        size_pred = 2.0 + raw_size  # Ensure minimum raise of 2BB
        
        return action_logits, size_pred 