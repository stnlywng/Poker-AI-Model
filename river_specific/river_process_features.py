import torch
import numpy as np

def process_static_features(row):
    """
    Process static features from the river gamestate.
    
    Args:
        row (dict): Dictionary containing:
            - hole_cards (list): List of 2 cards in format ['Ah', 'Kd']
            - flop_cards (list): List of 3 cards in format ['2h', '7d', 'Kc']
            - turn_card (str): Turn card in format '4s'
            - river_card (str): River card in format 'Jh'
            - position (str): Player position (e.g. 'bb', 'sb', 'utg1', etc.)
            - num_players (int): Number of players at the table
            - blinds (int): Big blind amount
            - ante (int): Ante amount
            - stacks (dict): Dictionary of player stacks
            - actions (list): List of actions in current round
            
    Returns:
        torch.Tensor: Tensor of shape (30,) containing:
            - 2 values for hole cards (rank and suit)
            - 6 values for flop cards (3 ranks and 3 suits)
            - 2 values for turn card (rank and suit)
            - 2 values for river card (rank and suit)
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
    for card in row['river_gamestate']['hole_cards']:
        static_features.append(rank_map[card[0]])
        static_features.append(suit_map[card[1]])

    # Flop cards
    for card in row['river_gamestate']['flop_cards']:
        static_features.append(rank_map[card[0]])
        static_features.append(suit_map[card[1]])

    # Turn card
    turn_card = row['river_gamestate']['turn_card']
    static_features.append(rank_map[turn_card[0]])
    static_features.append(suit_map[turn_card[1]])

    # River card
    river_card = row['river_gamestate']['river_card']
    static_features.append(rank_map[river_card[0]])
    static_features.append(suit_map[river_card[1]])

    # Num Players
    static_features.append(row['river_gamestate']['num_players'])

    # Position (one-hot encoded)
    positions = ['sb', 'bb', 'utg1', 'utg2', 'utg3', 'lj', 'hj', 'co', 'btn']
    position_oh = [1 if p == row['river_gamestate']['position'] else 0 for p in positions]
    static_features += position_oh

    # Numeric values (normalized)
    blinds = row['river_gamestate']['blinds']
    static_features.append(row['river_gamestate']['start_round_pot'] / blinds)

    # Player stacks
    player_stacks = {p['player']: p['stack'] for p in row['river_gamestate']['start_round_stacks']}

    # My stack (capped at 200 BB)
    my_stack = min(player_stacks[row['river_gamestate']['position']] / blinds, 200.0)
    static_features.append(my_stack)

    # Average stack (capped at 200 BB)
    avg_stack = min(sum(player_stacks.values()) / len(player_stacks) / blinds, 200.0)
    static_features.append(avg_stack)

    # Last-to-act stack (capped at 200 BB)
    actions_in_round = row['river_gamestate']['actions_in_round']
    if len(actions_in_round) > 0:
        last_to_act = actions_in_round[-1]['player']
        last_to_act_stack = min(player_stacks[last_to_act] / blinds, 200.0)
    else:
        # If no actions in river yet, take from turn
        turn_actions = row['river_gamestate']['turn_actions']
        for action in reversed(turn_actions):
            if action['player'] != row['river_gamestate']['position']:
                last_non_hero_player = action['player']
                last_to_act_stack = min(player_stacks[last_non_hero_player] / blinds, 200.0)
                break

    static_features.append(last_to_act_stack)

    # Current pot (sum of all post amounts)
    start_round_pot = row['river_gamestate'].get('start_round_pot', 0) or 0  # Convert None to 0
    action_sum = sum(float(a.get('amount', 0) or 0) for a in row['river_gamestate']['actions_in_round'])
    current_pot = start_round_pot + action_sum
    static_features.append(current_pot / blinds)

    return torch.FloatTensor(static_features)

def process_sequence_features(row):
    """
    Process sequence features from the river gamestate for GRU input.
    
    Args:
        row (dict): Dictionary containing river_gamestate with:
            - actions_in_round: List of actions in current round
            - preflop_actions: List of preflop actions
            - flop_actions: List of flop actions
            - turn_actions: List of turn actions
            - blinds: Big blind amount
            
    Returns:
        torch.FloatTensor: Tensor of shape (seq_len, 4) containing for each action:
            - action_type: Encoded action (0-7)
            - player_position: Encoded position (0-8)
            - normalized_amount: Amount normalized by big blind
            - round: 0 for preflop, 1 for flop, 2 for turn, 3 for river
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
    blinds = row['river_gamestate']['blinds']
    
    # Add preflop actions with round=0
    for action in row['river_gamestate']['preflop_actions']:
        amount = (action.get('amount', 0) or 0) / blinds
        action_type = history_action_map[action['action']]
        player_pos = player_map[action['player']]
        round_num = 0  # preflop
        
        sequence.append([action_type, player_pos, amount, round_num])
    
    # Add flop actions with round=1
    for action in row['river_gamestate']['flop_actions']:
        amount = (action.get('amount', 0) or 0) / blinds
        action_type = history_action_map[action['action']]
        player_pos = player_map[action['player']]
        round_num = 1  # flop
        
        sequence.append([action_type, player_pos, amount, round_num])
    
    # Add turn actions with round=2
    for action in row['river_gamestate']['turn_actions']:
        amount = (action.get('amount', 0) or 0) / blinds
        action_type = history_action_map[action['action']]
        player_pos = player_map[action['player']]
        round_num = 2  # turn
        
        sequence.append([action_type, player_pos, amount, round_num])

    # Add river actions with round=3
    for action in row['river_gamestate']['actions_in_round']:
        amount = (action.get('amount', 0) or 0) / blinds
        action_type = history_action_map[action['action']]
        player_pos = player_map[action['player']]
        round_num = 3  # river
        
        sequence.append([action_type, player_pos, amount, round_num])

    return torch.FloatTensor(sequence)

def process_features(row):
    """
    Process both static and sequence features from the river gamestate.
    
    Args:
        row (dict): Dictionary containing river_gamestate with all required fields
        
    Returns:
        tuple: (static_features, sequence_features)
            - static_features: torch.FloatTensor of shape (num_static_features,)
            - sequence_features: torch.FloatTensor of shape (seq_len, 4)
            
    Example input row structure:
    {
        'river_gamestate': {
            'hole_cards': ['Ah', 'Kd'],
            'flop_cards': ['2h', '7d', 'Kc'],
            'turn_card': '4s',
            'river_card': 'Jh',
            'num_players': 9,
            'position': 'btn',
            'blinds': 1.0,
            'start_round_pot': 1.5,
            'start_round_stacks': [{'player': 'sb', 'stack': 100}, ...],
            'actions_in_round': [
                {'player': 'sb', 'action': 'bets', 'amount': 1.0},
                ...
            ],
            'preflop_actions': [
                {'player': 'sb', 'action': 'posts', 'amount': 0.5},
                {'player': 'bb', 'action': 'posts', 'amount': 1.0},
                ...
            ],
            'flop_actions': [
                {'player': 'bb', 'action': 'checks', 'amount': 0},
                {'player': 'sb', 'action': 'bets', 'amount': 1.0},
                ...
            ],
            'turn_actions': [
                {'player': 'bb', 'action': 'checks', 'amount': 0},
                {'player': 'sb', 'action': 'bets', 'amount': 1.0},
                ...
            ]
        }
    }
    """
    static_features = process_static_features(row)
    sequence_features = process_sequence_features(row)
    
    return static_features, sequence_features 