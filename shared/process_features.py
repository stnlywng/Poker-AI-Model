import torch

def process_static_features(row, round):
    """
    Process static features from the gamestate.

    Args:
        row (dict): Dictionary containing:
            - hole_cards (list): List of 2 cards in format ['Ah', 'Kd'] - if applicable
            - flop_cards (list): List of 3 cards in format ['2h', '7d', 'Kc'] - if applicable
            - turn_card (str): Turn card in format '4s' - if applicable
            - river_card (str): River card in format 'Jh' - if applicable
            - position (str): Player position (e.g. 'bb', 'sb', 'utg1', etc.)
            - num_players (int): Number of players at the table
            - blinds (int): Big blind amount
            - ante (int): Ante amount
            - stacks (dict): Dictionary of player stacks
            - actions (list): List of actions in current round
        round (int): Round number (0: preflop, 1: flop, 2: turn, 3: river)

    Returns:
        torch.Tensor: Tensor of shape (19/25/27/29,) containing:
            - 2 values for hole cards (rank and suit)
            - 6 values for flop cards (3 ranks and 3 suits) - if applicable
            - 2 values for turn card (rank and suit) - if applicable
            - 2 values for river card (rank and suit) - if applicable
            - 1 value for num_players
            - 9 values for position one-hot encoding
            - 4 values for normalized numeric features (pot, stacks)
    """
    # Card processing
    rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13,
                'A': 14}
    suit_map = {'c': 0, 'd': 1, 'h': 2, 's': 3}

    static_features = []

    # Get the appropriate gamestate based on round
    gamestate = row['preflop_gamestate'] if round == 0 else row['flop_gamestate'] if round == 1 else row['turn_gamestate'] if round == 2 else row['river_gamestate']

    # Hole cards
    for card in gamestate['hole_cards']:
        static_features.append(rank_map[card[0]])
        static_features.append(suit_map[card[1]])

    # Flop cards
    if round >= 1:
        for card in gamestate['flop_cards']:
            static_features.append(rank_map[card[0]])
            static_features.append(suit_map[card[1]])

    # Turn card
    if round >= 2:
        turn_card = gamestate['turn_card']
        static_features.append(rank_map[turn_card[0]])
        static_features.append(suit_map[turn_card[1]])

    # River card
    if round >= 3:
        river_card = gamestate['river_card']
        static_features.append(rank_map[river_card[0]])
        static_features.append(suit_map[river_card[1]])

    # Num Players
    static_features.append(gamestate['num_players'])

    # Position (one-hot encoded)
    positions = ['sb', 'bb', 'utg1', 'utg2', 'utg3', 'lj', 'hj', 'co', 'btn']
    position_oh = [1 if p == gamestate['position'] else 0 for p in positions]
    static_features += position_oh

    # Numeric values (normalized)
    blinds = gamestate['blinds']
    static_features.append(gamestate['start_round_pot'] / blinds)

    # Player stacks
    player_stacks = {p['player']: p['stack'] for p in gamestate['start_round_stacks']}

    # My stack (capped at 200 BB)
    my_stack = min(player_stacks[gamestate['position']] / blinds, 200.0)
    static_features.append(my_stack)

    # Average stack (capped at 200 BB)
    avg_stack = min(sum(player_stacks.values()) / len(player_stacks) / blinds, 200.0)
    static_features.append(avg_stack)

    # Last-to-act stack (capped at 200 BB)
    actions_in_round = gamestate['actions_in_round']
    if len(actions_in_round) > 0:
        last_to_act = actions_in_round[-1]['player']
        last_to_act_stack = min(player_stacks[last_to_act] / blinds, 200.0)
    else:
        # If no actions in current round, take from previous round
        prev_round = round - 1
        prev_actions = gamestate['preflop_actions'] if prev_round == 0 else gamestate['flop_actions'] if prev_round == 1 else gamestate['turn_actions']
        for action in reversed(prev_actions):
            if action['player'] != gamestate['position']:
                last_non_hero_player = action['player']
                last_to_act_stack = min(player_stacks[last_non_hero_player] / blinds, 200.0)
                break

    static_features.append(last_to_act_stack)

    # Current pot (sum of all post amounts)
    start_round_pot = gamestate.get('start_round_pot', 0) or 0  # Convert None to 0
    action_sum = sum(float(a.get('amount', 0) or 0) for a in actions_in_round)
    current_pot = start_round_pot + action_sum
    static_features.append(current_pot / blinds)

    return torch.FloatTensor(static_features)

def process_sequence_features(row, round):
    """
    Process sequence features from the gamestate for GRU input.
    
    Args:
        row (dict): Dictionary containing gamestate with:
            - actions_in_round: List of actions in current round
            - preflop_actions: List of preflop actions (if applicable)
            - flop_actions: List of flop actions (if applicable)
            - turn_actions: List of turn actions (if applicable)
            - blinds: Big blind amount
        round (int): Round number (0: preflop, 1: flop, 2: turn, 3: river)
            
    Returns:
        torch.FloatTensor: Tensor of shape (seq_len, 4) containing for each action:
            - action_type: Encoded action (0-7)
            - player_position: Encoded position (0-8)
            - normalized_amount: Amount normalized by big blind
            - round: 0 for preflop, 1 for flop, 2 for turn, 3 for river
    """
    # Get the appropriate gamestate based on round
    gamestate = row['preflop_gamestate'] if round == 0 else row['flop_gamestate'] if round == 1 else row['turn_gamestate'] if round == 2 else row['river_gamestate']

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
    blinds = gamestate['blinds']

    if round >= 1:
        for action in gamestate['preflop_actions']:
            amount = (action.get('amount', 0) or 0) / blinds
            action_type = history_action_map[action['action']]
            player_pos = player_map[action['player']]
            sequence.append([action_type, player_pos, amount, 0])

    if round >= 2:
        for action in gamestate['flop_actions']:
            amount = (action.get('amount', 0) or 0) / blinds
            action_type = history_action_map[action['action']]
            player_pos = player_map[action['player']]
            sequence.append([action_type, player_pos, amount, 1])

    if round >= 3:
        for action in gamestate['turn_actions']:
            amount = (action.get('amount', 0) or 0) / blinds
            action_type = history_action_map[action['action']]
            player_pos = player_map[action['player']]
            sequence.append([action_type, player_pos, amount, 2])

    for action in gamestate['actions_in_round']:
        amount = (action.get('amount', 0) or 0) / blinds
        action_type = history_action_map[action['action']]
        player_pos = player_map[action['player']]
        sequence.append([action_type, player_pos, amount, round])

    return torch.FloatTensor(sequence)

def process_features(row, round):
    """
    Process both static and sequence features from the gamestate.
    
    Args:
        row (dict): Dictionary containing gamestate with all required fields
        round (int): Round number (0: preflop, 1: flop, 2: turn, 3: river)
        
    Returns:
        tuple: (static_features, sequence_features)
            - static_features: torch.FloatTensor of shape (num_static_features,)
            - sequence_features: torch.FloatTensor of shape (seq_len, 4)
    """
    static_features = process_static_features(row, round)
    sequence_features = process_sequence_features(row, round)
    
    return static_features, sequence_features 