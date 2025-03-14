def sample_one():
    hole_cards = ['Jh', 'Qs']
    position = 'lj'
    num_players = 7
    blinds = 300
    ante = 765

    stacks = {
        'sb': 1577,
        'bb': 1980,
        'utg1': 400,
        'lj': 1200,
        'hj': 1430,
        'co': 1150,
        'btn': 1304
    }

    actions = [
        {'action': 'folds', 'amount': 0, 'player': 'utg1'},
    ]

    return hole_cards, position, num_players, blinds, ante, stacks, actions

def sample_two():
    hole_cards = ['Kd', 'Qd']
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

    actions = [
        {'action': 'calls', 'amount': 800, 'player': 'utg1'},
        {'action': 'calls', 'amount': 800, 'player': 'utg2'},
        {'action': 'calls', 'amount': 800, 'player': 'lj'},
        {'action': 'calls', 'amount': 800, 'player': 'hj'},
        {'action': 'calls', 'amount': 800, 'player': 'co'},
        {'action': 'raises all-in', 'amount': 39000, 'player': 'btn'},
        {'action': 'calls all-in', 'amount': 19530, 'player': 'sb'},
    ]

    return hole_cards, position, num_players, blinds, ante, stacks, actions


def sample_three():
    hole_cards = ['Kd', 'Qd']
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

    actions = [
        {'action': 'calls', 'amount': 800, 'player': 'utg1'},
        {'action': 'calls', 'amount': 800, 'player': 'utg2'},
        {'action': 'calls', 'amount': 800, 'player': 'lj'},
        {'action': 'calls', 'amount': 800, 'player': 'hj'},
        {'action': 'calls', 'amount': 800, 'player': 'co'},
        {'action': 'raises all-in', 'amount': 39000, 'player': 'btn'},
        {'action': 'calls all-in', 'amount': 19530, 'player': 'sb'},
    ]

    return hole_cards, position, num_players, blinds, ante, stacks, actions