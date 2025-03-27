# Poker AI Model

A sophisticated deep learning model for poker decision-making, implementing a neural network architecture that handles different stages of the game (preflop, flop, turn, and river). The model uses advanced techniques including multi-head attention and bidirectional GRU to make poker decisions based on both static game state and action sequences.

## Architecture Overview

The project implements a neural network architecture with several key components:

- **Multi-Stage Processing**: Separate models for different poker stages:

  - Preflop
  - Flop
  - Turn
  - River

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
  - Handles static game features (19-dimensional)
  - Processes dynamic action sequences (4-dimensional per action)
- **Training Models**: Separate trained models for each game stage stored in `models/` directory

## Project Structure

poker-ai-model/
├── models/ # Trained model files
│ ├── poker_model_pf.pth
│ ├── poker_model_flop.pth
│ ├── poker_model_turn.pth
│ └── poker_model_river.pth
├── preflop/ # Preflop stage implementation
├── flop/ # Flop stage implementation
├── turn/ # Turn stage implementation
├── river/ # River stage implementation
├── shared/ # Shared utilities and functions
└── data/ # Training data and resources

## Technical Details

### Model Architecture

- **Input Processing**:

  - Static features: 19-dimensional input
  - Action sequences: 4-dimensional vectors (action_type, player, amount, round)
  - Sequence processing with variable length support

- **Network Components**:
  - Multi-head attention (8 heads)
  - 3-layer bidirectional GRU
  - Multiple batch normalization and dropout layers
  - Residual connections for better gradient flow

### Output Format

- Action probabilities for 6 possible actions
- Continuous bet/raise sizing predictions (minimum 2BB)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/poker-ai-model.git
cd poker-ai-model
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install torch numpy
```

## Usage

The model can be used for different stages of poker play:

```python
from preflop.pf_poker_model import PokerNet

# Initialize the model
model = PokerNet()

# Load trained weights
model.load_state_dict(torch.load('models/poker_model_pf.pth'))

# Make predictions
action_logits, size_pred = model(static_features, action_sequence, sequence_lengths)
```

## Training

Each stage has its own training script (e.g., `pf_train.py` for preflop) that handles:

- Data loading and preprocessing
- Model training with appropriate hyperparameters
- Model evaluation and saving

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Thanks to the PyTorch team for the excellent deep learning framework
- Special thanks to the poker community for valuable insights into game theory
