# Poker (No-Limit Holdem) AI Model

Poker No-Limit Holdem Model that learns how to play _exactly like you!_ Simply process and pipeline your hand history with my [PokerCraft Machine Learning Data Pre-Processor](https://github.com/stnlywng/Poker-ML-Data-Preprocessor). And then link your processed data file directories with our training modules.

## Personal Sample Usage.

I've trained data with poker professional/friend [KevinIsLaPoker](https://www.youtube.com/@%E5%87%AF%E6%96%87%E7%9A%84%E6%89%91%E5%85%8B%E4%B9%8B%E8%B7%AF) hand history (retrieved from PokerCraft), and acquired a model with 88% Validation Accuracy from his 170000+ data entries. I have then been able to give the model different scenarios using my test_models module, and retrieve impressive results.

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


