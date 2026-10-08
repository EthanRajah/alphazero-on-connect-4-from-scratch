# AlphaZero on Connect-4 from Scratch

Built a complete AlphaZero-style agent for Connect-4, from the bare game rules to a self-play training loop that pits a policy-value network guided by PUCT Monte Carlo Tree Search against itself. Implemented the board engine, neural network, MCTS, self-play data generation, training, and evaluation against baselines.

## How to run

```bash
python train.py --hidden_channels 16 --batch_size 8 --num_simulations 20 --num_games 10 --num_iterations 10 --num_epochs 10
```
