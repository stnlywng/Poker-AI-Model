import torch

def process_static_features(row):
    """
    Process static features from the flop gamestate.
    
    Args:
        row (dict): Dictionary containing flop_gamestate with:
            - hole_cards: List of 2 cards (e.g. ['Ah', 'Kd'])
            - flop_cards: List of 3 cards (e.g. ['2h', '7d', 'Kc'])
            - num_players: Number of players at table
            - position: Player position (sb, bb, utg1, etc.)
            - blinds: Big blind amount
            - start_round_pot: Initial pot
            - start_round_stacks: List of player stacks
            - actions_in_round: List of actions in current round
            
    Returns:
        torch.FloatTensor: Tensor of shape (num_static_features,) containing:
            - 4 values for hole cards (2 ranks + 2 suits)
            - 6 values for flop cards (3 ranks + 3 suits)
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
    for card in row['flop_gamestate']['hole_cards']:
        static_features.append(rank_map[card[0]])
        static_features.append(suit_map[card[1]])

    # Flop cards
    for card in row['flop_gamestate']['flop_cards']:
        static_features.append(rank_map[card[0]])
        static_features.append(suit_map[card[1]])

    # Num Players
    static_features.append(row['flop_gamestate']['num_players'])

    # Position (one-hot encoded)
    positions = ['sb', 'bb', 'utg1', 'utg2', 'utg3', 'lj', 'hj', 'co', 'btn']
    position_oh = [1 if p == row['flop_gamestate']['position'] else 0 for p in positions]
    static_features += position_oh

    # Numeric values (normalized)
    blinds = row['flop_gamestate']['blinds']
    static_features.append(row['flop_gamestate']['start_round_pot'] / blinds)

    # Player stacks
    player_stacks = {p['player']: p['stack'] for p in row['flop_gamestate']['start_round_stacks']}

    # My stack (capped at 200 BB)
    my_stack = min(player_stacks[row['flop_gamestate']['position']] / blinds, 200.0)
    static_features.append(my_stack)

    # Average stack (capped at 200 BB)
    avg_stack = min(sum(player_stacks.values()) / len(player_stacks) / blinds, 200.0)
    static_features.append(avg_stack)

    # Last-to-act stack (capped at 200 BB)

    # If no actions in flop yet (your first to act), take from preflop (that's not hero)
    actions_in_round = row['flop_gamestate']['actions_in_round']
    if len(actions_in_round) > 0:
        last_to_act = actions_in_round[-1]['player']
        last_to_act_stack = min(player_stacks[last_to_act] / blinds, 200.0)

    else:
        # print("ayo", row)
        preflop_actions = row['flop_gamestate']['preflop_actions']

        # Search preflop actions in reverse order
        for action in reversed(preflop_actions):
            if action['player'] != row['flop_gamestate']['position']:
                last_non_hero_player = action['player']
                last_to_act_stack = min(player_stacks[last_non_hero_player] / blinds, 200.0)
                # print(f"we found {last_non_hero_player} and {last_to_act_stack}")
                break
        # print("we found", )


    static_features.append(last_to_act_stack)

    # Current pot (sum of all post amounts)
    start_round_pot = row['flop_gamestate'].get('start_round_pot', 0) or 0  # Convert None to 0
    action_sum = sum(float(a.get('amount', 0) or 0) for a in row['flop_gamestate']['actions_in_round'])
    current_pot = start_round_pot + action_sum
    static_features.append(current_pot / blinds)

    return torch.FloatTensor(static_features)

def process_sequence_features(row):
    """
    Process sequence features from the flop gamestate for GRU input.
    
    Args:
        row (dict): Dictionary containing flop_gamestate with:
            - actions_in_round: List of actions in flop
            - preflop_actions: List of actions in preflop
            - blinds: Big blind amount
            
    Returns:
        torch.FloatTensor: Tensor of shape (seq_len, 4) containing for each action:
            - action_type: Encoded action (0-7)
            - player_position: Encoded position (0-8)
            - normalized_amount: Amount normalized by big blind
            - round: 0 for preflop, 1 for flop
    """
    # Sequence features (for GRU)
    history_action_map = {
        'posts': 0,  # Special action only in history
        'folds': 1,
        'calls': 2,  # Made consistent with target labels
        'raises': 3,  # Made consistent with target labels
        'calls all-in': 4,
        'raises all-in': 5,
        'checks': 6,
        'bets': 7,
        'bets all-in': 8
    }

    positions = ['sb', 'bb', 'utg1', 'utg2', 'utg3', 'lj', 'hj', 'co', 'btn']
    player_map = {p: i for i, p in enumerate(positions)}
    
    sequence = []
    blinds = row['flop_gamestate']['blinds']
    
    # Add preflop actions with round=0
    for action in row['flop_gamestate']['preflop_actions']:
        amount = (action.get('amount', 0) or 0) / blinds
        action_type = history_action_map[action['action']]
        player_pos = player_map[action['player']]
        round_num = 0  # preflop
        
        sequence.append([action_type, player_pos, amount, round_num])
    
    # Add flop actions with round=1
    for action in row['flop_gamestate']['actions_in_round']:
        amount = (action.get('amount', 0) or 0) / blinds
        action_type = history_action_map[action['action']]
        player_pos = player_map[action['player']]
        round_num = 1  # flop
        
        sequence.append([action_type, player_pos, amount, round_num])

    return torch.FloatTensor(sequence)

def process_features(row):
    """
    Process both static and sequence features from the flop gamestate.
    
    Args:
        row (dict): Dictionary containing flop_gamestate with all required fields
        
    Returns:
        tuple: (static_features, sequence_features)
            - static_features: torch.FloatTensor of shape (num_static_features,)
            - sequence_features: torch.FloatTensor of shape (seq_len, 4)
            
    Example input row structure:
    {
        "flop_gamestate": {
        "round": "f",
        "hole_cards": ["Tc", "7c"],
        "flop_cards": ["2s", "4c", "Jc"],
        "num_players": 7,
        "position": "btn",
        "blinds": 300,
        "start_round_pot": 2715,
        "start_round_stacks": [
          { "player": "lj", "stack": 10724 },
          ...
        ],
        "actions_in_round": [
          { "player": "sb", "action": "bets", "amount": 300 },
          ...
        ],
        "preflop_actions": [
          { "player": "lj", "action": "posts", "amount": 45 },
          { "player": "sb", "action": "posts", "amount": 45 },
          ...
        ]
    }
    """
    static_features = process_static_features(row)
    sequence_features = process_sequence_features(row)
    
    return static_features, sequence_features
