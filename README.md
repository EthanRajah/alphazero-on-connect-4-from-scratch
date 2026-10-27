# AlphaZero on Connect-4 from Scratch

Built a complete AlphaZero-style agent for Connect-4, from the bare game rules to a self-play training loop that pits a policy-value network guided by PUCT Monte Carlo Tree Search against itself. Implemented the board engine, neural network, MCTS, self-play data generation, training, and evaluation against baselines.

https://github.com/user-attachments/assets/3d019cde-f949-4f2a-a09d-2ae04a66226a

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

Connect 4 is a two-player game played on a 6-row, 7-column grid, with players alternating turns dropping pieces into columns. The first player to connect four of their pieces in a row (horizontally, vertically, or diagonally) wins. If the board fills up without a winner, the game is a draw. This can be modelled as a zero-sum game, making it suitable for reinforcement learning.

Following the framework of AlphaZero, we can use Monte Carlo Tree Search (MCTS) to simulate games and determine the best actions that lead to a win. MCTS has 3 main steps:
1. **Selection**: Starting from the root node, select child nodes based on a balance of exploration and exploitation until reaching a leaf node.
2. **Expansion & Evaluation**: If the leaf node is not terminal (winning/losing/drawing state), expand it by adding all possible child nodes (legal moves). Evaluate the new leaf nodes to estimate their value in terms of how likely they are to lead to a win.
3. **Backpropagation**: Update the value estimates of all nodes along the path from the leaf node back to the root based on the evaluation results.

A couple of key questions arise when attempting to apply MCTS to Connect 4:
1. How do we actually evaluate the leaf nodes to determine their value? Do we need to simulate the game to the end and then make some estimate for that node? That could get really expensive if we abstracted this to a more complex game like chess or go.
2. With the wide range of possible moves and state space, how do we ensure that our MCTS is efficient and doesn't waste time exploring unpromising branches of the tree? I'd want to have my algorithm be able to beat me as fast as possible, and that means making sure it doesn't even consider bad moves.


