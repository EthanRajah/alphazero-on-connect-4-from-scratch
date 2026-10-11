# AlphaZero on Connect-4 from Scratch

Built a complete AlphaZero-style agent for Connect-4, from the bare game rules to a self-play training loop that pits a policy-value network guided by PUCT Monte Carlo Tree Search against itself. Implemented the board engine, neural network, MCTS, self-play data generation, training, and evaluation against baselines.

## How to run

```bash
python train.py --hidden_channels 16 --batch_size 8 --num_simulations 20 --num_games 10 --num_iterations 10 --num_epochs 10
```

To play against the included trained network with the graphical interface:

```bash
python play_connect4.py
```

The board accepts mouse clicks or the number keys `1` through `7`. Pieces
animate into place, and the network's selected column and policy probabilities
are shown while it decides.

After training a different network, pass its checkpoint and architecture:

```bash
python play_connect4.py --checkpoint path/to/network.pth --hidden_channels 16 --num_simulations 8 
```

The graphical evaluator is also available from `train.py --evaluate_mode human`
after a training run.

## Underlying Concepts

