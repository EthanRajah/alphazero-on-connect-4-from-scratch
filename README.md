# AlphaZero on Connect-4 from Scratch

Built a complete AlphaZero-style agent for Connect-4, from the bare game rules to a self-play training loop that pits a policy-value network guided by PUCT Monte Carlo Tree Search against itself. Implemented the board engine, neural network, MCTS, self-play data generation, training, and evaluation against baselines.

![Connect4 gameplay](assets/connect4_demo.gif)

*Network playing (and winning) against a human expert (me lol), with 50 MCTS simulation steps, and the policy decision distribution shown after each move.*

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
python play_connect4.py --checkpoint path/to/network.pth --hidden_channels 16 --num_simulations 100 
```

The graphical evaluator is also available from `train.py --evaluate_mode human`
after a training run.

## Underlying Concepts

Connect 4 is a two-player game played on a 6-row, 7-column grid, with players alternating turns dropping pieces into columns. The first player to connect four of their pieces in a row (horizontally, vertically, or diagonally) wins. If the board fills up without a winner, the game is a draw. This can be modelled as a zero-sum game, making it suitable for reinforcement learning.

Following the framework of AlphaZero, we can use Monte Carlo Tree Search (MCTS) to simulate games and determine the best actions that lead to a win. MCTS has 3 main steps:
1. **Selection**: Starting from the root node, select child nodes based on a balance of exploration and exploitation until reaching a leaf node. Nodes are scored using both factors, written as Score(s, a) = Q(s, a) + U(s, a), where Q is the expected reward of the node and U is an exploration term that encourages promising or less-visited nodes to be explored.
2. **Expansion & Evaluation**: If the leaf node is not terminal (winning/losing/drawing state), expand it by adding all possible child nodes (legal moves). Evaluate the new leaf nodes to estimate their value in terms of how likely they are to lead to a win.
3. **Backpropagation**: Update the value estimates of all nodes along the path from the leaf node back to the root based on the evaluation results.

A couple of key questions arise when attempting to apply MCTS to Connect 4:
1. How do we actually evaluate the leaf nodes to determine their value? Do we need to simulate the game to the end and then make some estimate for that node? That could get really expensive if we abstracted this to a more complex game like chess or go where there are way more possible moves.
2. With the wide range of possible moves and state space, how do we ensure that our MCTS is efficient and doesn't waste time exploring unpromising branches of the tree? I'd want to have my algorithm be able to beat me as fast as possible, and that means making sure it doesn't even consider bad moves.

The **Predictor Upper Confidence Bounds applied to Trees (PUCT)** algorithm is a solution to both of these problems. Building on the scoring function from MCTS for the selection step, PUCT selects a child action a at state s that maximizes the following score:

$$Score(s, a) = Q(s, a) + U(s, a)$$
$$Score(s, a) = Q(s, a) + c_{puct} * P(s, a) * \frac{\sqrt{N(s)}}{1 + N(s, a)}$$

Where:
- Q(s, a) is the expected reward of taking action a at state s, based on the average of the values of all simulations that have passed through this node.
- U(s, a) is an exploration term that encourages the selection of less-visited nodes, where c_puct is a hyperparameter that controls the balance between exploration and exploitation, P(s, a) is the prior probability of taking action a at state s, and N(s) is the total number of times the parent node has been visited, while N(s, a) is the number of times action a has been taken from state s.

To compute Q(s, a), we can train a value network that takes the current board state as input and outputs a scalar value representing the expected outcome of the game from that state (-1 for a loss, 0 for a draw, and +1 for a win). Regular L2 loss can be used to train the value network, comparing predicted values to the actual values obtained by simulated self-play games (tracking the outcome of each game and assigning the appropriate value target to each state visited during the game).

To compute P(s, a), we can train a policy network that takes the current board state as input and outputs a probability distribution over all possible actions (for our case, columns 0-6 to drop a piece into). The policy network can be trained using cross-entropy loss, comparing predicted distribution to the actual distribution obtained by computing the visit count distribution of actions from the MCTS simulations per state. Thus, the visit count distribution can be normalized to form a probability distribution, which serves as the target for the policy network.

One can observe that the policy network is improved with more accurate visit count distributions from MCTS, while the value network is improved with more accurate value targets from self-play games. This creates a feedback loop where the policy and value networks improve each other over time, leading to stronger play. Hence, these networks are combined into a single policy-value network that outputs both the action probabilities and the value estimate for a given board state.

## Network Architecture

While further testing and tuning of the network architecture is possible, my current best implementation uses a small convolutional neural network backbone with 4 convolutional layers, each followed by a ReLU activation function. The latents from the backbone are then fed into two seperate MLPs, one for the policy head, outputting a probability distribution over the 7 possible actions, and one for the value head, outputting a scalar value representing the expected outcome of the game from that state.

![Network architecture diagram](assets/network_diagram.png)

## Experiments and Evaluation

