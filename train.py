"""
AlphaZero on Connect-4 from Scratch

Training script for long-running self-play training and hyperparameter tuning.

Run this with: python train.py
Uses functions defined in model.py.
"""

import argparse

from model import *
import numpy as np
import torch

import matplotlib.pyplot as plt

def train_connect4(hidden_channels, batch_size, num_simulations, c_puct, temperature, num_games, num_iterations, num_epochs, evaluate_mode="baseline"):
    """Train a policy-value network on Connect-4 using self play and MCTS"""
    net = build_policy_value_net(in_channels=2, hidden_channels=hidden_channels, num_columns=7)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    training_history = train_loop(net, optimizer, num_iterations, num_games, num_simulations, c_puct, batch_size, num_epochs=num_epochs, temperature=temperature)
    plot_training_history(training_history)
    evaluate_connect4(net, num_matches=5, seed=0, mode=evaluate_mode)

def plot_training_history(training_history):
    """Plot the total loss over the training iterations."""
    total_losses = [np.mean([epoch_loss["total"] for epoch_loss in iteration["losses"]]) for iteration in training_history]
    plt.plot(total_losses)
    plt.xlabel("Training Iteration")
    plt.ylabel("Average Total Loss")
    plt.title("Training Loss Over Iterations")
    plt.grid()
    plt.show()

def evaluate_connect4(net, num_matches=1, seed=0, mode="baseline"):
    """Evaluate the trained policy-value network against a random player. Baseline will use uniform random moves, while human will allow a human player to play against the trained network."""
    if mode == "baseline":
        print("Starting baseline evaluation against random player...")
        evaluate_against_random(net, num_matches, seed)
    elif mode == "human":
        print("Starting human vs trained network evaluation...")
        evaluate_against_human(net, num_matches)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_channels", type=int, default=8, help="Number of channels in the hidden layers of the policy-value network.")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of steps from the rollout to use per gradient descent step.")
    parser.add_argument("--num_simulations", type=int, default=8, help="Number of MCTS simulations per move.")
    parser.add_argument("--c_puct", type=float, default=1.5, help="Exploration constant for MCTS.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature parameter for action selection.")
    parser.add_argument("--num_games", type=int, default=2, help="Number of self-play games to generate.")
    parser.add_argument("--num_iterations", type=int, default=10, help="Number of training iterations as a whole.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of iterations of the collected rollout to run.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--evaluate_mode", type=str, default="baseline", choices=["baseline", "human"], help="Evaluation mode: 'baseline' for random player, 'human' for human player.")
    args = parser.parse_args()

    print("==== AlphaZero on Connect-4 Training ===")
    print("Training Connect-4 with the following parameters:")
    print(f"Hidden channels: {args.hidden_channels}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of simulations: {args.num_simulations}")
    print(f"c_puct: {args.c_puct}")
    print(f"Temperature: {args.temperature}")
    print(f"Number of self-play games: {args.num_games}")
    print(f"Number of training iterations: {args.num_iterations}")
    print(f"Number of epochs per iteration: {args.num_epochs}")
    print("==== Starting Training ===")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_connect4(
        hidden_channels=args.hidden_channels,
        batch_size=args.batch_size,
        num_simulations=args.num_simulations,
        c_puct=args.c_puct,
        temperature=args.temperature,
        num_games=args.num_games,
        num_iterations=args.num_iterations,
        num_epochs=args.num_epochs,
        evaluate_mode=args.evaluate_mode
    )

if __name__ == "__main__":
    main()