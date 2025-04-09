# Poker (No-Limit Holdem) AI Model

Given a No-Limit Holdem Spot, the Poker No-Limit Holdem Model will tell you what they think should be done based on their training (from Poker Hand History).

## Usage and Setup
1. Download your No-Limit Holdem' hand histories from PokerCraft.
2. Process and pipeline your hand history with my [PokerCraft Machine Learning Data Pre-Processor](https://github.com/stnlywng/Poker-ML-Data-Preprocessor). This tool allows you to pre-process the data into Parquet Format, so that we can effectively train our model.
3. Train the models (one for each stage of the game):
  ```bash
  python pf_train.py <data_path> <epochs> <model_save_path> <batch_size>
  
  python flop_train.py <data_path> <epochs> <model_save_path> <batch_size>
  
  python turn_train.py <data_path> <epochs> <model_save_path> <batch_size>
  
  python river_train.py <data_path> <epochs> <model_save_path> <batch_size>
  
- `<data_path>` — Required location of your data (e.g., ./data/preflop)
- `<epochs>` — Optional, number of training epochs (default is 20)
- `<model_save_path>` — Required location of where to store your model, make sure location exists beforehand (e.g., ./models) 
- `<batch_size>` — Optional (default is 32) 
  ```
5. Run test_models where you give a spot in json format (input), and get a resulting output.

## Sample.

I've trained data with poker professional/friend [KevinIsLaPoker](https://www.youtube.com/@%E5%87%AF%E6%96%87%E7%9A%84%E6%89%91%E5%85%8B%E4%B9%8B%E8%B7%AF) hand history (retrieved from PokerCraft), and acquired a model with 88% Validation Accuracy from his 170000+ data entries. 

**Sample Input**:
```json
{
  "preflop_gamestate": {
    "round": "pf",
    "hole_cards": [
      "Ad",
      "Kd"
    ],
    "num_players": 6,
    "position": "sb",
    "blinds": 100,
    "start_round_pot": 150,
    "start_round_stacks": [
      {
        "player": "hj",
        "stack": 17872
      },
      {
        "player": "co",
        "stack": 26018
      },
      {
        "player": "btn",
        "stack": 10000
      },
      {
        "player": "sb",
        "stack": 11858
      },
      {
        "player": "bb",
        "stack": 11382
      },
      {
        "player": "utg1",
        "stack": 10000
      }
    ],
    "actions_in_round": [
      {
        "player": "sb",
        "action": "posts",
        "amount": 50
      },
      {
        "player": "bb",
        "action": "posts",
        "amount": 100
      },
      {
        "player": "utg1",
        "action": "folds"
      },
      {
        "player": "hj",
        "action": "folds"
      },
      {
        "player": "co",
        "action": "folds"
      },
      {
        "player": "btn",
        "action": "raises",
        "amount": 220
      }
    ]
  },
  "label": {
    "action": "raises",
    "size": 1100
  }
}
```
**Output:**
```text

=== Model Prediction ===
Recommended Action: raise
Recommended Raise Size: 8.50 BB

Action Probabilities:
fold: 15.06%
call: 20.00%
raise: 64.04%
call all-in: 0.12%
raise all-in: 0.28%
check: 0.50%
```

## Architecture Overview

The project implements a neural network architecture with several key components:

- **Multi-Stage Processing**: Separate models for different Poker Stages (Preflop, Flop, Turn, River).

- **Neural Network Architecture**:
  - Multi-head attention mechanism for processing action sequences
  - Bidirectional GRU for temporal pattern recognition
  - Deep static feature processing
  - Dual output heads for action selection and sizing decisions

### Key Features

- **Advanced Attention Mechanism**: Uses multi-head attention to focus on relevant historical actions
- **Bidirectional Processing**: Implements 3-layer bidirectional GRU for comprehensive sequence understanding
- **Dual Output System**:
  - Action head for decision making (fold, call, raise)
  - Size head for bet/raise sizing with minimum 2BB enforcement
- **Robust Feature Processing**:
  - Handles static game features (PF - 19, FLP - 25, TRN - 27, RIV - 29)
  - Processes dynamic action sequences (4-dimensional per action)
- **Training Models**: Separate trained models for each game stage stored in `models/` directory


