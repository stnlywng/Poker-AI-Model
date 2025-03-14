import torch

def process_static_features(row):
    """
    Process static features from the preflop gamestate.
    
    Args:
        row (dict): Dictionary containing preflop_gamestate with:
            - hole_cards: List of 2 cards (e.g. ['Ah', 'Kd'])
            - num_players: Number of players at table
            - position: Player position (sb, bb, utg1, etc.)
            - blinds: Big blind amount
            - start_round_pot: Initial pot
            - start_round_stacks: List of player stacks
            - actions_in_round: List of actions in current round
            
    Returns:
        torch.FloatTensor: Tensor of shape (num_static_features,) containing:
            - 4 values for hole cards (2 ranks + 2 suits)
            - 1 value for num_players
            - 9 values for position one-hot encoding
            - 4 values for normalized numeric features (pot, stacks)
    """
    # Card processing
    rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13,
                'A': 14}
    suit_map = {'c': 0, 'd': 1, 'h': 2, 's': 3}

    static_features = []

    # Hole cards
    for card in row['preflop_gamestate']['hole_cards']:
        static_features.append(rank_map[card[0]])
        static_features.append(suit_map[card[1]])

    # Num Players
    static_features.append(row['preflop_gamestate']['num_players'])

    # Position (one-hot encoded)
    positions = ['sb', 'bb', 'utg1', 'utg2', 'utg3', 'lj', 'hj', 'co', 'btn']
    position_oh = [1 if p == row['preflop_gamestate']['position'] else 0 for p in positions]
    static_features += position_oh

    # Numeric values (normalized)
    blinds = row['preflop_gamestate']['blinds']
    static_features.append(row['preflop_gamestate']['start_round_pot'] / blinds)

    # Player stacks
    player_stacks = {p['player']: p['stack'] for p in row['preflop_gamestate']['start_round_stacks']}

    # My stack (capped at 200 BB)
    my_stack = min(player_stacks[row['preflop_gamestate']['position']] / blinds, 200.0)
    static_features.append(my_stack)

    # Average stack (capped at 200 BB)
    avg_stack = min(sum(player_stacks.values()) / len(player_stacks) / blinds, 200.0)
    static_features.append(avg_stack)

    # Last-to-act stack (capped at 200 BB)
    last_to_act = row['preflop_gamestate']['actions_in_round'][-1]['player']
    last_to_act_stack = min(player_stacks[last_to_act] / blinds, 200.0)
    static_features.append(last_to_act_stack)

    # Current pot (sum of all post amounts)
    start_round_pot = row['preflop_gamestate'].get('start_round_pot', 0) or 0  # Convert None to 0
    action_sum = sum(float(a.get('amount', 0) or 0) for a in row['preflop_gamestate']['actions_in_round'])
    current_pot = start_round_pot + action_sum
    static_features.append(current_pot / blinds)

    return torch.FloatTensor(static_features)

def process_sequence_features(row):
    """
    Process sequence features from the preflop gamestate for GRU input.
    
    Args:
        row (dict): Dictionary containing preflop_gamestate with:
            - actions_in_round: List of actions in current round
            - blinds: Big blind amount
            
    Returns:
        torch.FloatTensor: Tensor of shape (seq_len, 3) containing for each action:
            - action_type: Encoded action (0-6)
            - player_position: Encoded position (0-8)
            - normalized_amount: Amount normalized by big blind
    """
    # Sequence features (for GRU)
    history_action_map = {
        'posts': 0,  # Special action only in history
        'folds': 1,
        'calls': 2,  # Made consistent with target labels
        'raises': 3,  # Made consistent with target labels
        'calls all-in': 4,
        'raises all-in': 5,
        'checks': 6
    }

    positions = ['sb', 'bb', 'utg1', 'utg2', 'utg3', 'lj', 'hj', 'co', 'btn']
    player_map = {p: i for i, p in enumerate(positions)}
    
    sequence = []
    blinds = row['preflop_gamestate']['blinds']
    
    for action in row['preflop_gamestate']['actions_in_round']:
        # Normalize amount by big blind
        amount = (action.get('amount', 0) or 0) / blinds
        action_type = history_action_map[action['action']]
        player_pos = player_map[action['player']]
        
        sequence.append([action_type, player_pos, amount])

    return torch.FloatTensor(sequence)

def process_features(row):
    """
    Process both static and sequence features from the preflop gamestate.
    
    Args:
        row (dict): Dictionary containing preflop_gamestate with all required fields
        
    Returns:
        tuple: (static_features, sequence_features)
            - static_features: torch.FloatTensor of shape (num_static_features,)
            - sequence_features: torch.FloatTensor of shape (seq_len, 3)
            
    Example input row structure:
    {
        'preflop_gamestate': {
            'hole_cards': ['Ah', 'Kd'],
            'num_players': 9,
            'position': 'btn',
            'blinds': 1.0,
            'start_round_pot': 1.5,
            'start_round_stacks': [{'player': 'sb', 'stack': 100}, ...],
            'actions_in_round': [
                {'player': 'sb', 'action': 'posts', 'amount': 0.5},
                {'player': 'bb', 'action': 'posts', 'amount': 1.0},
                {'player': 'utg1', 'action': 'folds', 'amount': 0},
                ...
            ]
        }
    }
    """
    static_features = process_static_features(row)
    sequence_features = process_sequence_features(row)
    
    return static_features, sequence_features
