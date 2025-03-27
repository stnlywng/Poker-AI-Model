def sample_zero():
    hole_cards = ['Qh', 'Ts']
    flop_cards = ['2h', '7d', 'Kc']
    turn_card = '4s'
    position = 'lj'
    num_players = 7
    blinds = 50
    ante = 60

    stacks = {
        'sb': 1577,
        'bb': 1980,
        'utg1': 400,
        'lj': 1200,
        'hj': 1430,
        'co': 1150,
        'btn': 1304
    }

    preflop_actions = [
        {'action': 'posts', 'amount': 25, 'player': 'sb'},
        {'action': 'posts', 'amount': 50, 'player': 'bb'},
        {'action': 'folds', 'amount': 0, 'player': 'utg1'},
        {'action': 'raises', 'amount': 150, 'player': 'lj'},
        {'action': 'folds', 'amount': 0, 'player': 'hj'},
        {'action': 'folds', 'amount': 0, 'player': 'co'},
        {'action': 'folds', 'amount': 0, 'player': 'btn'},
        {'action': 'folds', 'amount': 0, 'player': 'sb'},
        {'action': 'calls', 'amount': 100, 'player': 'bb'}
    ]

    flop_actions = [
        {'action': 'checks', 'amount': 0, 'player': 'bb'},
        {'action': 'bets', 'amount': 100, 'player': 'lj'},
        {'action': 'calls', 'amount': 100, 'player': 'bb'}
    ]

    actions = [
        {'action': 'checks', 'amount': 0, 'player': 'bb'},
    ]

    return hole_cards, flop_cards, turn_card, position, num_players, blinds, ante, stacks, preflop_actions, flop_actions, actions

def sample_one():
    hole_cards = ['Ah', 'Kd']
    flop_cards = ['2h', '7d', 'Kc']
    turn_card = '4s'
    position = 'co'
    num_players = 8
    blinds = 100
    ante = 0

    stacks = {
        'sb': 10000,
        'bb': 10000,
        'utg1': 10000,
        'utg2': 10000,
        'lj': 10000,
        'hj': 10000,
        'co': 10000,
        'btn': 10000
    }

    preflop_actions = [
        {'action': 'posts', 'amount': 50, 'player': 'sb'},
        {'action': 'posts', 'amount': 100, 'player': 'bb'},
        {'action': 'folds', 'amount': 0, 'player': 'utg1'},
        {'action': 'folds', 'amount': 0, 'player': 'utg2'},
        {'action': 'folds', 'amount': 0, 'player': 'lj'},
        {'action': 'folds', 'amount': 0, 'player': 'hj'},
        {'action': 'raises', 'amount': 300, 'player': 'co'},
        {'action': 'folds', 'amount': 0, 'player': 'btn'},
        {'action': 'folds', 'amount': 0, 'player': 'sb'},
        {'action': 'calls', 'amount': 200, 'player': 'bb'}
    ]

    flop_actions = [
        {'action': 'checks', 'amount': 0, 'player': 'bb'},
        {'action': 'bets', 'amount': 200, 'player': 'co'},
        {'action': 'calls', 'amount': 200, 'player': 'bb'}
    ]

    actions = [
        {'action': 'checks', 'amount': 0, 'player': 'bb'},
    ]

    return hole_cards, flop_cards, turn_card, position, num_players, blinds, ante, stacks, preflop_actions, flop_actions, actions

def sample_two():
    hole_cards = ['Ad', 'Ac']
    flop_cards = ['2h', '7d', 'Kc']
    turn_card = '4s'
    position = 'bb'
    num_players = 8
    blinds = 800
    ante = 840

    stacks = {
        'sb': 19530,
        'bb': 89900,
        'utg1': 19500,
        'utg2': 15300,
        'lj': 26900,
        'hj': 19100,
        'co': 19000,
        'btn': 39000
    }

    preflop_actions = [
        {'action': 'posts', 'amount': 400, 'player': 'sb'},
        {'action': 'posts', 'amount': 800, 'player': 'bb'},
        {'action': 'calls', 'amount': 800, 'player': 'utg1'},
        {'action': 'calls', 'amount': 800, 'player': 'utg2'},
        {'action': 'calls', 'amount': 800, 'player': 'lj'},
        {'action': 'calls', 'amount': 800, 'player': 'hj'},
        {'action': 'calls', 'amount': 800, 'player': 'co'},
        {'action': 'raises', 'amount': 1700, 'player': 'btn'},
        {'action': 'raises', 'amount': 4800, 'player': 'sb'},
        {'action': 'calls', 'amount': 4000, 'player': 'bb'},
        {'action': 'calls', 'amount': 3100, 'player': 'utg1'},
        {'action': 'calls', 'amount': 3100, 'player': 'utg2'},
        {'action': 'calls', 'amount': 3100, 'player': 'lj'},
        {'action': 'calls', 'amount': 3100, 'player': 'hj'},
        {'action': 'calls', 'amount': 3100, 'player': 'co'},
        {'action': 'calls', 'amount': 2200, 'player': 'btn'}
    ]

    flop_actions = [
        {'action': 'bets', 'amount': 4000, 'player': 'sb'},
        {'action': 'calls', 'amount': 4000, 'player': 'bb'},
        {'action': 'calls', 'amount': 4000, 'player': 'utg1'},
        {'action': 'calls', 'amount': 4000, 'player': 'utg2'},
        {'action': 'calls', 'amount': 4000, 'player': 'lj'},
        {'action': 'calls', 'amount': 4000, 'player': 'hj'},
        {'action': 'calls', 'amount': 4000, 'player': 'co'},
        {'action': 'calls', 'amount': 4000, 'player': 'btn'}
    ]

    actions = [
        {'action': 'bets', 'amount': 8000, 'player': 'sb'},
    ]

    return hole_cards, flop_cards, turn_card, position, num_players, blinds, ante, stacks, preflop_actions, flop_actions, actions

def sample_three():
    hole_cards = ['Kd', 'Qd']
    flop_cards = ['2h', '7d', 'Kc']
    turn_card = '4s'
    position = 'bb'
    num_players = 8
    blinds = 800
    ante = 840

    stacks = {
        'sb': 19530,
        'bb': 89900,
        'utg1': 19500,
        'utg2': 15300,
        'lj': 26900,
        'hj': 19100,
        'co': 19000,
        'btn': 39000
    }

    preflop_actions = [
        {'action': 'posts', 'amount': 400, 'player': 'sb'},
        {'action': 'posts', 'amount': 800, 'player': 'bb'},
        {'action': 'calls', 'amount': 800, 'player': 'utg1'},
        {'action': 'calls', 'amount': 800, 'player': 'utg2'},
        {'action': 'calls', 'amount': 800, 'player': 'lj'},
        {'action': 'calls', 'amount': 800, 'player': 'hj'},
        {'action': 'calls', 'amount': 800, 'player': 'co'},
        {'action': 'raises all-in', 'amount': 39000, 'player': 'btn'},
        {'action': 'calls all-in', 'amount': 19530, 'player': 'sb'},
        {'action': 'calls', 'amount': 38200, 'player': 'bb'},
        {'action': 'calls', 'amount': 18700, 'player': 'utg1'},
        {'action': 'calls', 'amount': 14500, 'player': 'utg2'},
        {'action': 'calls', 'amount': 26100, 'player': 'lj'},
        {'action': 'calls', 'amount': 18300, 'player': 'hj'},
        {'action': 'calls', 'amount': 18200, 'player': 'co'}
    ]

    flop_actions = [
        {'action': 'bets', 'amount': 8000, 'player': 'sb'},
        {'action': 'calls', 'amount': 8000, 'player': 'bb'},
        {'action': 'calls', 'amount': 8000, 'player': 'utg1'},
        {'action': 'calls', 'amount': 8000, 'player': 'utg2'},
        {'action': 'calls', 'amount': 8000, 'player': 'lj'},
        {'action': 'calls', 'amount': 8000, 'player': 'hj'},
        {'action': 'calls', 'amount': 8000, 'player': 'co'}
    ]

    actions = [
        {'action': 'bets', 'amount': 16000, 'player': 'sb'},
    ]

    return hole_cards, flop_cards, turn_card, position, num_players, blinds, ante, stacks, preflop_actions, flop_actions, actions 