import torch
from flop_poker_model import PokerNet
from flop_process_features import process_features
from flop_testing_samples import *


def load_model(model_path='../models/poker_model_flop.pth'):
    """Load the trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PokerNet(
        static_dim=25,  # From process_features
        action_dim=4,  # [action_type, player, amount, round]
        hidden_dim=256,  # Increased from 128
        gru_hidden_dim=128  # Increased from 64
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

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

def create_sample_gamestate(hole_cards, flop_cards, position, num_players, blinds, stacks, ante, preflop_actions, actions):
    """Create a sample gamestate dictionary matching the format expected by process_features"""
    positions = get_positions_for_size(num_players)
    
    # Calculate initial pot (antes + blinds)
    start_round_pot = blinds * 1.5  # SB + BB
    if ante > 0:
        start_round_pot += ante
    
    # Create automatic ante and blind actions
    auto_actions = []
    if ante > 0:
        for pos in positions:
            auto_actions.append({
                'player': pos,
                'action': 'posts',
                'amount': ante / num_players
            })
    
    # Add blind posts
    auto_actions.extend([
        {
            'player': 'sb',
            'action': 'posts',
            'amount': blinds // 2
        },
        {
            'player': 'bb',
            'action': 'posts',
            'amount': blinds
        }
    ])
    
    # Add preflop actions
    auto_actions.extend(preflop_actions)
    
    return {
        'flop_gamestate': {
            'hole_cards': hole_cards,
            'flop_cards': flop_cards,
            'position': position,
            'num_players': num_players,
            'blinds': blinds,
            'start_round_pot': start_round_pot,
            'start_round_stacks': [
                {'player': pos, 'stack': stacks[pos]} for pos in positions
            ],
            'actions_in_round': actions,
            'preflop_actions': auto_actions
        }
    }

def predict_action(model, gamestate, device):
    """Make a prediction using the model"""
    # Process features
    static_features, action_sequence = process_features(gamestate)
    
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
    # Load the model
    print("Loading model...")
    model, device = load_model()
    print("Model loaded successfully!")
    
    while True:
        print("\n=== Test the Poker Model ===")
        print("Enter the following information (or 'q' to quit):")
        
        # Get input from user
        if input("\nReady to enter a hand? (y/q): ").lower() == 'q':
            break

        hole_cards, flop_cards, position, num_players, blinds, ante, stacks, preflop_actions, actions = sample_three()
        
        # Create gamestate
        gamestate = create_sample_gamestate(
            hole_cards=hole_cards,
            flop_cards=flop_cards,
            position=position,
            num_players=num_players,
            blinds=blinds,
            ante=ante,
            stacks=stacks,
            preflop_actions=preflop_actions,
            actions=actions
        )
        
        # Get prediction
        result = predict_action(model, gamestate, device)
        
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