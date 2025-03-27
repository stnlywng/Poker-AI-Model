import torch
from preflop_specific.pf_poker_model import PokerNet as PokerNetPF
from flop_poker_model import PokerNet
from shared.process_features import process_features
import json
import sys

PREFLOP_MODEL = '../models/poker_model_pf_multihead.pth'
FLOP_MODEL = '../models/poker_model_flop.pth'
TURN_MODEL = '../models/poker_model_turn.pth'
RIVER_MODEL = '../models/poker_model_river.pth'

def load_model(round):
    if round == 0:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = PokerNetPF(
            static_dim=19,  # From process_features
            action_dim=4,  # [action_type, player, amount, round]
            hidden_dim=256,  # Increased from 128
            gru_hidden_dim=128  # Increased from 64
        )
        model.load_state_dict(torch.load(PREFLOP_MODEL, map_location=device))
        model.to(device)
        model.eval()
        return model, device
    elif round == 1:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = PokerNet(
            static_dim=25,  # From process_features
            action_dim=4,  # [action_type, player, amount, round]
            hidden_dim=256,  # Increased from 128
            gru_hidden_dim=128  # Increased from 64
        )
        model.load_state_dict(torch.load(FLOP_MODEL, map_location=device))
        model.to(device)
        model.eval()
        return model, device
    elif round == 2:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = PokerNet(
            static_dim=27,  # From process_features
            action_dim=4,  # [action_type, player, amount, round]
            hidden_dim=256,  # Increased from 128
            gru_hidden_dim=128  # Increased from 64
        )   
        model.load_state_dict(torch.load(TURN_MODEL, map_location=device))
        model.to(device)
        model.eval()
        return model, device
    elif round == 3:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = PokerNet(
            static_dim=29,  # From process_features
            action_dim=4,  # [action_type, player, amount, round]
            hidden_dim=256,  # Increased from 128
            gru_hidden_dim=128  # Increased from 64
        )
        model.load_state_dict(torch.load(RIVER_MODEL, map_location=device))
        model.to(device)
        model.eval()
        return model, device
    else:
        raise Exception("Invalid Round Number")

def get_positions_for_size(num_players):
    """Get the correct position names based on table size"""
    if num_players == 9:
        return ['sb', 'bb', 'utg1', 'utg2', 'utg3', 'lj', 'hj', 'co', 'btn']
    elif num_players == 8:
        return ['sb', 'bb', 'utg1', 'utg2', 'lj', 'hj', 'co', 'btn']
    elif num_players == 7:
        return ['sb', 'bb', 'utg1', 'lj', 'hj', 'co', 'btn']
    elif num_players == 6:
        return ['sb', 'bb', 'utg1', 'hj', 'co', 'btn']
    elif num_players == 5:
        return ['sb', 'bb', 'hj', 'co', 'btn']
    elif num_players == 4:
        return ['sb', 'bb', 'co', 'btn']
    elif num_players == 3:
        return ['sb', 'bb', 'btn']
    else:  # heads up
        return ['sb', 'bb']

def predict_action(model, gamestate, round, device):
    """Make a prediction using the model"""
    # Process features
    static_features, action_sequence = process_features(gamestate, round)
    
    print("\nModel Input:")
    print(f"Static features shape: {static_features.shape}")
    print(f"Action sequence shape: {action_sequence.shape}")
    
    # Add batch dimension and move to device
    static_features = static_features.unsqueeze(0).to(device)
    action_sequence = action_sequence.unsqueeze(0).to(device)
    sequence_length = torch.tensor([len(action_sequence[0])]).to(device)
    
    # Get prediction
    with torch.no_grad():
        action_logits, size_pred = model(static_features, action_sequence, sequence_length)
        print(f"\nRaw logits: {action_logits}")
        action_probs = torch.softmax(action_logits, dim=1)
        predicted_action = torch.argmax(action_logits, dim=1).item()
        
    # Map predictions back to human-readable format
    action_map = {
        0: 'fold',
        1: 'call',
        2: 'raise',
        3: 'call all-in',
        4: 'raise all-in',
        5: 'check',
        6: 'bet',
        7: 'bet all-in'
    }
    
    # Get probabilities for each action
    probs_dict = {action_map[i]: prob.item() for i, prob in enumerate(action_probs[0])}
    
    return {
        'predicted_action': action_map[predicted_action],
        'action_probabilities': probs_dict,
        'raise_size': size_pred.item() if predicted_action in [2, 4, 6] else None
    }

def main():
    json_path = '../json-samples/river.json'
    
    # Read JSON file
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        sys.exit(1)
    
    # Determine round from JSON data
    if 'river_gamestate' in data:
        round_num = 3  # River
        print("Detected: River round")
    elif 'turn_gamestate' in data:
        round_num = 2  # Turn
        print("Detected: Turn round")
    elif 'flop_gamestate' in data:
        round_num = 1  # Flop
        print("Detected: Flop round")
    else:
        round_num = 0  # Preflop
        print("Detected: Preflop round")
    
    # Load appropriate model
    print(f"\nLoading model for round {round_num}...")
    model, device = load_model(round_num)
    print("Model loaded successfully!")
    
    # Create gamestate from JSON data
    gamestate = data  # The JSON data is already in the correct format
    
    # Get prediction
    print("\nMaking prediction...")
    result = predict_action(model, gamestate, round_num, device)
    
    # Display results
    print("\n=== Model Prediction ===")
    print(f"Recommended Action: {result['predicted_action']}")
    if result['raise_size'] is not None:
        print(f"Recommended Raise Size: {result['raise_size']:.2f} BB")
    
    print("\nAction Probabilities:")
    for action, prob in result['action_probabilities'].items():
        print(f"{action}: {prob:.2%}")

if __name__ == "__main__":
    main() 